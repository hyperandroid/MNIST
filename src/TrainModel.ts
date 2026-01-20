import {GPUEnv} from "./GPUEnv";
import {KernelRegistry} from "./tensor/kernel/KernelRegistry";
import {TensorManager} from "./tensor/TensorManager";
import {MNISTDatasource, MNISTDataSourceIterator} from "./MNIST/MNISTDatasource";
import {Sequential} from "./layer/Sequential";
import {Linear} from "./layer/Linear";
import {heNormal} from "./math/Utils";
import {ReLU} from "./layer/ReLU";
import {Dropout} from "./layer/Dropout";

/**
 * Initialize the GPU. If not present, no training will be possible.
 */
await GPUEnv.init()

const tm = new TensorManager(GPUEnv.device);
const kernelRegistry = new KernelRegistry(GPUEnv.device, tm);


/**
 * test code
 */

const M = 4096, K = 4096, N = 128;

const aSize = M * K;
const bSize = K * N;

// Initialize input tensors
const A = new Float32Array(aSize);
const B = new Float32Array(bSize);

// Simple deterministic values (good for debugging)
for (let i = 0; i < A.length; i++) A[i] = (i % 97) * 0.01 * (Math.random()<.5 ? 1 : -1);
for (let i = 0; i < B.length; i++) B[i] = (i % 89) * 0.02;

const t0 = tm.getTensorBuffer(
	"t0",
	GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	[M,K],
	A,
);
const t1 = tm.getTensorBuffer(
	"t1",
	GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	[K,N],
	B,
);

const out = kernelRegistry.relu.run(
	kernelRegistry.matmul.run(t0, t1)
);

const output2 = await tm.readBuffer(out.buffer, out.sizeInBytes());
console.log(output2);


const sm = tm.getTensorBuffer(
	"softmax",
	GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
	[1,10],
	new Float32Array([0,1,2,3,4,5,6,7,8,9]),
);

const smout = kernelRegistry.softmax.run(sm);
const smout2 = await tm.readBuffer(smout.buffer, smout.sizeInBytes());
console.log(smout2);

const datasource = new MNISTDatasource();
await datasource
	.load()
	.catch((e: Error) => {
		throw new Error("Failed to load data source: " + e)
	});

function it(iterator: MNISTDataSourceIterator) {
	const data = iterator.next();
	for (let i = 0; i < data.size; i++) {
		MNISTDatasource.ShowRandomImage(data.data.subarray(
			28 * 28 * i,
			28 * 28 * (1 + i),
		));
		console.log(data.labels[i]);
	}
}
it(datasource.getTrainIterator(4))
it(datasource.getTestIterator(4))



const linear = new Sequential(
	new Linear(tm, kernelRegistry, {
		name: "first",
		inputFeatures: 768,
		outputFeatures: 256,
		useBias: true,
		initializer: heNormal
	}),
	new ReLU(tm, kernelRegistry, "ReLU1"),
	new Dropout(tm, kernelRegistry, "dropout1", 0.5),

	new Linear(tm, kernelRegistry, {
		name: "second",
		inputFeatures: 256,
		outputFeatures: 128,
		useBias: true,
		initializer: heNormal
	}),
	new ReLU(tm, kernelRegistry, "ReLU2"),
	new Dropout(tm, kernelRegistry, "dropout2", 0.3),

	new Linear(tm, kernelRegistry, {
		name: "third",
		inputFeatures: 128,
		outputFeatures: 10,
		useBias: true,
		initializer: heNormal
	}),

)
