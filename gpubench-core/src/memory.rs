//! GPU buffer management — allocation, upload, download.
//!
//! Uses raw Vulkan memory management (vkAllocateMemory / vkBindBufferMemory)
//! Uses raw Vulkan memory management (vkAllocateMemory / vkBindBufferMemory).

use ash::vk;

use crate::vulkan::VulkanContext;

// ═══════════════════════════════════════════════════════════════════════════════
// GpuBuffer
// ═══════════════════════════════════════════════════════════════════════════════

/// A GPU buffer with bound device memory.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

impl GpuBuffer {
    /// Create a new buffer with the given usage and memory properties.
    pub fn new(
        ctx: &VulkanContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        mem_properties: vk::MemoryPropertyFlags,
    ) -> anyhow::Result<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { ctx.device.create_buffer(&buffer_info, None)? };
        let mem_req = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };

        let mem_type = ctx
            .find_memory_type(mem_req.memory_type_bits, mem_properties)
            .ok_or_else(|| {
                // Clean up buffer on failure
                unsafe { ctx.device.destroy_buffer(buffer, None) };
                anyhow::anyhow!(
                    "No suitable memory type for size={}, flags={:?}",
                    size,
                    mem_properties
                )
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type);

        let memory = unsafe {
            ctx.device.allocate_memory(&alloc_info, None).map_err(|e| {
                ctx.device.destroy_buffer(buffer, None);
                anyhow::anyhow!("Memory allocation failed: {:?}", e)
            })?
        };

        unsafe {
            ctx.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| {
                    ctx.device.free_memory(memory, None);
                    ctx.device.destroy_buffer(buffer, None);
                    anyhow::anyhow!("Bind buffer memory failed: {:?}", e)
                })?;
        }

        Ok(Self {
            buffer,
            memory,
            size,
        })
    }

    /// Create a device-local buffer (fast GPU memory, not host-visible).
    pub fn device_local(
        ctx: &VulkanContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> anyhow::Result<Self> {
        Self::new(
            ctx,
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    }

    /// Create a host-visible staging buffer (for CPU ↔ GPU transfers).
    pub fn staging(ctx: &VulkanContext, size: vk::DeviceSize) -> anyhow::Result<Self> {
        Self::new(
            ctx,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    /// Destroy the buffer and free its memory.
    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_buffer(self.buffer, None);
            ctx.device.free_memory(self.memory, None);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Data Transfer Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Upload f32 data to a device-local buffer via a staging buffer.
pub fn upload_f32(
    ctx: &VulkanContext,
    dst: &GpuBuffer,
    data: &[f32],
) -> anyhow::Result<()> {
    let byte_size = (data.len() * std::mem::size_of::<f32>()) as vk::DeviceSize;
    let staging = GpuBuffer::staging(ctx, byte_size)?;

    // Map staging, copy data
    unsafe {
        let ptr = ctx
            .device
            .map_memory(staging.memory, 0, byte_size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, byte_size as usize);
        ctx.device.unmap_memory(staging.memory);
    }

    // Copy staging → device
    ctx.execute_commands(|cb| unsafe {
        let copy = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: byte_size,
        };
        ctx.device
            .cmd_copy_buffer(cb, staging.buffer, dst.buffer, &[copy]);
    })?;

    staging.destroy(ctx);
    Ok(())
}

/// Download f32 data from a device-local buffer via a staging buffer.
pub fn download_f32(
    ctx: &VulkanContext,
    src: &GpuBuffer,
    count: usize,
) -> anyhow::Result<Vec<f32>> {
    let byte_size = (count * std::mem::size_of::<f32>()) as vk::DeviceSize;
    let staging = GpuBuffer::staging(ctx, byte_size)?;

    // Copy device → staging
    ctx.execute_commands(|cb| unsafe {
        let copy = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: byte_size,
        };
        ctx.device
            .cmd_copy_buffer(cb, src.buffer, staging.buffer, &[copy]);
    })?;

    // Map staging, read data
    let data = unsafe {
        let ptr = ctx
            .device
            .map_memory(staging.memory, 0, byte_size, vk::MemoryMapFlags::empty())?;
        let slice = std::slice::from_raw_parts(ptr as *const f32, count);
        let vec = slice.to_vec();
        ctx.device.unmap_memory(staging.memory);
        vec
    };

    staging.destroy(ctx);
    Ok(data)
}

/// Read a single f32 value from a buffer at a byte offset.
pub fn read_f32_at(
    ctx: &VulkanContext,
    src: &GpuBuffer,
    byte_offset: vk::DeviceSize,
) -> anyhow::Result<f32> {
    let staging = GpuBuffer::staging(ctx, 4)?;

    ctx.execute_commands(|cb| unsafe {
        let copy = vk::BufferCopy {
            src_offset: byte_offset,
            dst_offset: 0,
            size: 4,
        };
        ctx.device
            .cmd_copy_buffer(cb, src.buffer, staging.buffer, &[copy]);
    })?;

    let value = unsafe {
        let ptr = ctx
            .device
            .map_memory(staging.memory, 0, 4, vk::MemoryMapFlags::empty())?;
        let val = *(ptr as *const f32);
        ctx.device.unmap_memory(staging.memory);
        val
    };

    staging.destroy(ctx);
    Ok(value)
}

/// Read a single u32 value from a buffer at a byte offset.
pub fn read_u32_at(
    ctx: &VulkanContext,
    src: &GpuBuffer,
    byte_offset: vk::DeviceSize,
) -> anyhow::Result<u32> {
    let staging = GpuBuffer::staging(ctx, 4)?;

    ctx.execute_commands(|cb| unsafe {
        let copy = vk::BufferCopy {
            src_offset: byte_offset,
            dst_offset: 0,
            size: 4,
        };
        ctx.device
            .cmd_copy_buffer(cb, src.buffer, staging.buffer, &[copy]);
    })?;

    let value = unsafe {
        let ptr = ctx
            .device
            .map_memory(staging.memory, 0, 4, vk::MemoryMapFlags::empty())?;
        let val = *(ptr as *const u32);
        ctx.device.unmap_memory(staging.memory);
        val
    };

    staging.destroy(ctx);
    Ok(value)
}
