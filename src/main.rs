#![feature(const_option, new_uninit, test)]

#[cfg(test)]
mod tests {
    extern crate test;

    use std::sync::Arc;
    use vulkano::{
        buffer::{
            allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
            BufferUsage, Subbuffer,
        },
        command_buffer::{
            allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        },
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator,
            layout::{
                DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
                DescriptorType,
            },
            PersistentDescriptorSet, WriteDescriptorSet,
        },
        device::{
            physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features,
            Queue, QueueCreateInfo, QueueFlags,
        },
        instance::{Instance, InstanceCreateInfo},
        memory::allocator::{DeviceLayout, MemoryUsage, StandardMemoryAllocator},
        pipeline::{
            layout::PipelineLayoutCreateInfo, ComputePipeline, PipelineBindPoint, PipelineLayout,
        },
        shader::ShaderStages,
        DeviceSize, NonZeroDeviceSize, VulkanLibrary,
    };

    /// Prebaked (and hence deterministic) randomness without duplicates.
    static SHUFFLE_LUT: [u8; 0x100] = [
        188, 221, 178, 138, 25, 69, 108, 147, 255, 238, 70, 208, 171, 219, 214, 40, 162, 251, 237,
        33, 133, 107, 229, 34, 227, 168, 146, 10, 88, 236, 29, 209, 47, 48, 35, 224, 43, 204, 96,
        109, 184, 98, 63, 95, 26, 207, 194, 66, 37, 140, 235, 135, 0, 223, 58, 177, 81, 116, 55,
        89, 139, 250, 129, 7, 74, 228, 215, 121, 196, 148, 102, 125, 122, 113, 94, 28, 216, 142,
        59, 182, 203, 198, 41, 175, 156, 128, 92, 38, 186, 197, 65, 99, 170, 13, 155, 202, 193,
        212, 114, 118, 226, 87, 11, 112, 249, 60, 185, 90, 201, 181, 14, 199, 2, 54, 187, 73, 150,
        165, 153, 120, 130, 179, 127, 134, 222, 189, 158, 152, 154, 17, 123, 12, 53, 45, 51, 61,
        166, 79, 44, 3, 86, 234, 57, 52, 31, 119, 240, 42, 231, 19, 4, 100, 67, 145, 248, 220, 172,
        245, 124, 160, 1, 82, 85, 32, 241, 49, 22, 254, 72, 9, 24, 36, 247, 117, 246, 205, 243,
        157, 106, 183, 230, 84, 137, 163, 149, 161, 242, 104, 75, 144, 18, 80, 46, 191, 143, 93,
        78, 71, 141, 56, 225, 164, 252, 244, 39, 217, 5, 213, 8, 77, 62, 239, 64, 210, 23, 97, 30,
        192, 253, 151, 16, 21, 218, 101, 167, 190, 206, 103, 200, 174, 76, 232, 68, 110, 159, 15,
        131, 211, 83, 50, 27, 195, 91, 111, 126, 180, 115, 105, 169, 6, 20, 233, 176, 173, 132,
        136,
    ];
    const SHUFFLE_ALG: ShuffleAlg = ShuffleAlg::Single;

    /// Must be large enough for the choice of `ShuffleAlg`, or there are going to be out-of-bounds
    /// panics inside `create_buffers`.
    const BUFFERS: usize = 0x10000;

    fn shuffle(i: usize) -> usize {
        match SHUFFLE_ALG {
            ShuffleAlg::None => i,
            ShuffleAlg::Single => (i & !0xFF) | SHUFFLE_LUT[i & 0xFF] as usize,
            ShuffleAlg::Double => {
                (i & !0xFFFF)
                    | ((SHUFFLE_LUT[i & 0x00FF] as usize) << 8)
                    | SHUFFLE_LUT[(i & 0xFF00) >> 8] as usize
            }
        }
    }

    #[allow(unused)]
    enum ShuffleAlg {
        None,
        Single,
        Double,
    }

    #[bench]
    fn dispatch_indirect(bencher: &mut test::Bencher) {
        let (device, queue) = setup();
        let buffers = create_buffers(
            &device,
            DeviceLayout::from_size_alignment(12, 4).unwrap(),
            BufferUsage::INDIRECT_BUFFER,
        );
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: r"
                    #version 460

                    void main() {}
                ",
            }
        }

        let shader = cs::load(device.clone()).unwrap();
        let pipeline = ComputePipeline::new(
            device,
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .unwrap();

        bencher.iter(|| {
            let mut builder = AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder.bind_pipeline_compute(pipeline.clone());

            for buffer in &*buffers {
                builder
                    // SAFETY: We created the buffers with a size of 12 and alignment of 4.
                    .dispatch_indirect(unsafe { buffer.reinterpret_ref_unchecked() }.clone())
                    .unwrap();
            }

            builder
        });
    }

    #[bench]
    fn descriptor_set(bencher: &mut test::Bencher) {
        let (device, queue) = setup();
        let buffers = create_buffers(
            &device,
            DeviceLayout::new(
                NonZeroDeviceSize::MIN,
                // Make sure all the subbuffers fit in the same buffer.
                device
                    .physical_device()
                    .properties()
                    .min_storage_buffer_offset_alignment,
            )
            .unwrap(),
            BufferUsage::STORAGE_BUFFER,
        );
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: r"
                    #version 460
                    #extension GL_EXT_nonuniform_qualifier : enable

                    layout(set = 0, binding = 0) buffer Buf {
                        int x;
                    } buffers[];

                    layout(push_constant) uniform PC {
                        uint index;
                    };

                    void main() {
                        buffers[index].x = 0;
                    }
                ",
            }
        }

        let shader = cs::load(device.clone()).unwrap();
        let descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: [(
                    0,
                    DescriptorSetLayoutBinding {
                        descriptor_count: BUFFERS as u32,
                        variable_descriptor_count: true,
                        stages: ShaderStages::COMPUTE,
                        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
                    },
                )]
                .into(),
                ..Default::default()
            },
        )
        .unwrap();
        let pipeline_layout = {
            PipelineLayout::new(
                device.clone(),
                PipelineLayoutCreateInfo {
                    set_layouts: vec![descriptor_set_layout.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let pipeline = ComputePipeline::with_pipeline_layout(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            pipeline_layout.clone(),
            None,
        )
        .unwrap();
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device);
        let descriptor_set = PersistentDescriptorSet::new_variable(
            &descriptor_set_allocator,
            descriptor_set_layout,
            BUFFERS as u32,
            [WriteDescriptorSet::buffer_array(
                0,
                0,
                buffers.iter().cloned(),
            )],
        )
        .unwrap();

        bencher.iter(|| {
            let mut builder = AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder.bind_pipeline_compute(pipeline.clone());
            builder.bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline_layout.clone(),
                0,
                descriptor_set.clone(),
            );
            builder.dispatch([1, 1, 1]).unwrap();

            builder
        });
    }

    fn setup() -> (Arc<Device>, Arc<Queue>) {
        let library = VulkanLibrary::new().unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(_, q)| q.queue_flags.intersects(QueueFlags::GRAPHICS))
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: Features {
                    descriptor_binding_variable_descriptor_count: true,
                    runtime_descriptor_array: true,
                    ..Features::empty()
                },
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        (device, queues.next().unwrap())
    }

    fn create_buffers(
        device: &Arc<Device>,
        layout: DeviceLayout,
        buffer_usage: BufferUsage,
    ) -> Box<[Subbuffer<[u8]>]> {
        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
        let buffer_allocator = SubbufferAllocator::new(
            memory_allocator,
            SubbufferAllocatorCreateInfo {
                arena_size: BUFFERS as DeviceSize * layout.pad_to_alignment().size(),
                buffer_usage,
                memory_usage: MemoryUsage::DeviceOnly, // avoids atom size
                ..Default::default()
            },
        );
        let mut buffers = Box::new_uninit_slice(BUFFERS);
        for i in 0..BUFFERS {
            buffers[shuffle(i)].write(buffer_allocator.allocate(layout).unwrap());
        }

        // SAFETY: The shuffle LUT has no duplicates, meaning that any mapping based on it that
        // doesn't discard any indices forms an isomorphic mapping. `shuffle` doesn't discard any
        // indices, and we feed it all indices in the range [0, BUFFERS), which means that all
        // indices must have been written to.
        unsafe { Box::<[_]>::assume_init(buffers) }
    }
}

fn main() {}
