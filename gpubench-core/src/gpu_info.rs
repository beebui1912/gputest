//! GPU discovery and enumeration.

use ash::vk;

use crate::vulkan::{self, VulkanInstance};

// ═══════════════════════════════════════════════════════════════════════════════
// GpuInfo
// ═══════════════════════════════════════════════════════════════════════════════

/// Information about a discovered GPU.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub device_type: String,
    pub vendor: String,
    pub vendor_id: u32,
    pub driver_version: String,
    pub api_version: String,
    pub dedicated_vram_mb: Option<u64>,
    pub device_index: usize,
    pub physical_device: vk::PhysicalDevice,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Enumeration
// ═══════════════════════════════════════════════════════════════════════════════

/// Enumerate all Vulkan-capable GPUs on the system.
pub fn enumerate_gpus(instance: &VulkanInstance) -> anyhow::Result<Vec<GpuInfo>> {
    let physical_devices = instance.enumerate_physical_devices()?;
    let mut gpus = Vec::with_capacity(physical_devices.len());

    for (i, &pd) in physical_devices.iter().enumerate() {
        let props = instance.get_properties(pd);
        let mem_props = instance.get_memory_properties(pd);

        // Find the largest DEVICE_LOCAL heap — this is the dedicated VRAM
        let mut max_device_local_bytes: u64 = 0;
        for heap_idx in 0..mem_props.memory_heap_count as usize {
            let heap = mem_props.memory_heaps[heap_idx];
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                max_device_local_bytes = max_device_local_bytes.max(heap.size);
            }
        }

        let dedicated_vram_mb = if max_device_local_bytes > 0 {
            Some(max_device_local_bytes / (1024 * 1024))
        } else {
            None
        };

        gpus.push(GpuInfo {
            name: vulkan::vk_string(&props.device_name),
            device_type: vulkan::device_type_str(props.device_type).to_string(),
            vendor: vulkan::vendor_name(props.vendor_id).to_string(),
            vendor_id: props.vendor_id,
            driver_version: vulkan::format_driver_version(props.vendor_id, props.driver_version),
            api_version: format!(
                "{}.{}.{}",
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version)
            ),
            dedicated_vram_mb,
            device_index: i,
            physical_device: pd,
        });
    }

    Ok(gpus)
}
