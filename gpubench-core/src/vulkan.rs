//! Vulkan instance and device context management.
//!
//! Provides two-phase initialization:
//! 1. `VulkanInstance` — creates Vulkan entry + instance for GPU enumeration
//! 2. `VulkanContext` — creates logical device, compute queue, and command pool
//!    for a specific physical device

use ash::vk;
use std::ffi::CStr;

use crate::compute::Shaders;

// ═══════════════════════════════════════════════════════════════════════════════
// VulkanInstance — Phase 1: enumeration only
// ═══════════════════════════════════════════════════════════════════════════════

pub struct VulkanInstance {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    /// Set to `true` inside [`Self::create_context`] to disable double-destroy
    /// from `Drop` when ownership has been transferred into a [`VulkanContext`].
    ownership_transferred: bool,
}

impl VulkanInstance {
    /// Create a Vulkan instance for GPU enumeration.
    pub fn new() -> anyhow::Result<Self> {
        let entry = unsafe {
            ash::Entry::load()
                .map_err(|e| anyhow::anyhow!("Failed to load Vulkan loader: {}", e))?
        };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"gpubench")
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"gpubench-core")
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_2);

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|e| anyhow::anyhow!("Failed to create Vulkan instance: {:?}", e))?
        };

        Ok(Self {
            entry,
            instance,
            ownership_transferred: false,
        })
    }

    /// Enumerate all physical devices.
    pub fn enumerate_physical_devices(&self) -> anyhow::Result<Vec<vk::PhysicalDevice>> {
        unsafe {
            self.instance
                .enumerate_physical_devices()
                .map_err(|e| anyhow::anyhow!("Failed to enumerate physical devices: {:?}", e))
        }
    }

    /// Get physical device properties.
    pub fn get_properties(
        &self,
        pd: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceProperties {
        unsafe { self.instance.get_physical_device_properties(pd) }
    }

    /// Get physical device memory properties.
    pub fn get_memory_properties(
        &self,
        pd: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceMemoryProperties {
        unsafe { self.instance.get_physical_device_memory_properties(pd) }
    }

    /// Get physical device queue family properties.
    pub fn get_queue_family_properties(
        &self,
        pd: vk::PhysicalDevice,
    ) -> Vec<vk::QueueFamilyProperties> {
        unsafe {
            self.instance
                .get_physical_device_queue_family_properties(pd)
        }
    }

    /// Consume the instance to create a full VulkanContext for a specific device.
    /// This compiles all compute shaders and creates the command pool.
    ///
    /// After a successful call, the original `VulkanInstance`'s `Drop` becomes
    /// a no-op — ownership of the underlying Vulkan instance handle is now the
    /// `VulkanContext`'s responsibility.
    pub fn create_context(
        mut self,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<VulkanContext> {
        // Find compute queue family
        let queue_families = unsafe {
            self.instance
                .get_physical_device_queue_family_properties(physical_device)
        };

        let compute_queue_family = queue_families
            .iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| anyhow::anyhow!("No compute queue family found"))?
            as u32;

        // Create logical device
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&queue_priorities);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));

        let device = unsafe {
            self.instance
                .create_device(physical_device, &device_create_info, None)?
        };

        let compute_queue = unsafe { device.get_device_queue(compute_queue_family, 0) };

        // Create command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };

        // Query properties
        let memory_properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(physical_device)
        };
        let device_properties = unsafe {
            self.instance
                .get_physical_device_properties(physical_device)
        };

        // Compile all compute shaders
        let shaders = Shaders::new(&device)?;

        // Ownership transferred — clone the Arc-backed wrappers, then mark
        // `self` so its `Drop` skips destroy_instance.
        let entry = self.entry.clone();
        let instance = self.instance.clone();
        self.ownership_transferred = true;

        Ok(VulkanContext {
            entry,
            instance,
            physical_device,
            device,
            compute_queue,
            compute_queue_family,
            command_pool,
            memory_properties,
            device_properties,
            shaders,
        })
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        if !self.ownership_transferred {
            unsafe {
                self.instance.destroy_instance(None);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VulkanContext — Phase 2: full device context for benchmarks
// ═══════════════════════════════════════════════════════════════════════════════

pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub compute_queue: vk::Queue,
    pub compute_queue_family: u32,
    pub command_pool: vk::CommandPool,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device_properties: vk::PhysicalDeviceProperties,
    pub shaders: Shaders,
}

impl VulkanContext {
    /// Execute a one-shot command buffer: record → submit → wait for fence.
    pub fn execute_commands<F>(&self, record: F) -> anyhow::Result<()>
    where
        F: FnOnce(vk::CommandBuffer),
    {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cb = unsafe { self.device.allocate_command_buffers(&alloc_info)? }[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device.begin_command_buffer(cb, &begin_info)?;
        }

        record(cb);

        unsafe {
            self.device.end_command_buffer(cb)?;
        }

        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };

        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));

        unsafe {
            self.device
                .queue_submit(self.compute_queue, &[submit_info], fence)?;
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.destroy_fence(fence, None);
            self.device
                .free_command_buffers(self.command_pool, &[cb]);
        }

        Ok(())
    }

    /// Insert a buffer memory barrier into a command buffer.
    pub fn cmd_buffer_barrier(
        &self,
        cb: vk::CommandBuffer,
        buffer: vk::Buffer,
        size: vk::DeviceSize,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
    ) {
        let barrier = vk::BufferMemoryBarrier::default()
            .buffer(buffer)
            .offset(0)
            .size(size)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cb,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }
    }

    /// Find a memory type that satisfies both the type filter and property flags.
    pub fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && self.memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return Some(i);
            }
        }
        None
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.shaders.destroy(&self.device);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Decode Vulkan vendor ID to human-readable name.
pub fn vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        0x1002 => "AMD",
        0x10DE => "NVIDIA",
        0x8086 => "Intel",
        0x13B5 => "ARM",
        0x5143 => "Qualcomm",
        0x1010 => "ImgTec",
        _ => "Unknown",
    }
}

/// Format driver version based on vendor conventions.
pub fn format_driver_version(vendor_id: u32, driver_version: u32) -> String {
    match vendor_id {
        // NVIDIA: 10-bit major, 8-bit minor, 14-bit patch
        0x10DE => format!(
            "{}.{}.{}",
            (driver_version >> 22) & 0x3FF,
            (driver_version >> 14) & 0xFF,
            driver_version & 0x3FFF
        ),
        // Intel: 14-bit major, 14-bit minor
        0x8086 => format!("{}.{}", driver_version >> 14, driver_version & 0x3FFF),
        // Standard Vulkan version encoding
        _ => format!(
            "{}.{}.{}",
            vk::api_version_major(driver_version),
            vk::api_version_minor(driver_version),
            vk::api_version_patch(driver_version)
        ),
    }
}

/// Extract a Rust String from a Vulkan fixed-size C string array.
pub fn vk_string(raw: &[std::ffi::c_char]) -> String {
    unsafe {
        CStr::from_ptr(raw.as_ptr())
            .to_string_lossy()
            .to_string()
    }
}

/// Format Vulkan device type enum.
pub fn device_type_str(dt: vk::PhysicalDeviceType) -> &'static str {
    match dt {
        vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
        vk::PhysicalDeviceType::CPU => "CPU (Software Rendering)",
        _ => "Unknown",
    }
}
