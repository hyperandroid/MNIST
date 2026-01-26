(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))s(a);new MutationObserver(a=>{for(const t of a)if(t.type==="childList")for(const i of t.addedNodes)i.tagName==="LINK"&&i.rel==="modulepreload"&&s(i)}).observe(document,{childList:!0,subtree:!0});function r(a){const t={};return a.integrity&&(t.integrity=a.integrity),a.referrerPolicy&&(t.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?t.credentials="include":a.crossOrigin==="anonymous"?t.credentials="omit":t.credentials="same-origin",t}function s(a){if(a.ep)return;a.ep=!0;const t=r(a);fetch(a.href,t)}})();class N{static device;static async init(){if(!navigator.gpu)throw new Error("WebGPU not available in this browser/context.");const e=await navigator.gpu.requestAdapter();if(!e)throw new Error("Failed to get GPU adapter.");N.device=await e.requestDevice()}}function d(c,e){return Math.floor((c+e-1)/e)}class p{constructor(e,r,s){this.kr=s,this.module=e.createShaderModule({code:r}),this.pipeline=e.createComputePipeline({layout:"auto",compute:{module:this.module,entryPoint:"main"}})}pipeline;module}class R{constructor(e,r){this.savedTensors=e,this.kr=r}name="MatMulBackward";backward(e){const[r,s]=this.savedTensors;let a=null;if(r.requiresGradient){const i=this.kr.transpose.run(s);a=this.kr.matmul.run(e,i)}let t=null;if(s.requiresGradient){const i=this.kr.transpose.run(r);t=this.kr.matmul.run(i,e)}return[a,t]}}class l extends p{constructor(e,r,s){super(e,l.matmulWGSL,s),this.device=e,this.tm=r,this.kr=s,this.paramsBuf=e.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(4);paramsBuf;run(e,r,s){if(e.shape.length!==2||r.shape.length!==2||s!==void 0&&s.shape.length!==2)throw new Error("MatMul: expected 2D tensors");if(e.shape[1]!==r.shape[0])throw new Error("MatMul: invalid dimensions");const a=e.shape[0],t=e.shape[1],i=r.shape[1];if(s=s??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[a,i]),s.shape[0]!==a||s.shape[1]!==i)throw new Error("MatMul: invalid output dimensions");this.params[0]=a,this.params[1]=i,this.params[2]=t,this.device.queue.writeBuffer(this.paramsBuf,0,this.params);const n=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:s.buffer}},{binding:3,resource:{buffer:this.paramsBuf}}]}),o=d(i,16),u=d(a,16),f=this.device.createCommandEncoder(),h=f.beginComputePass();return h.setPipeline(this.pipeline),h.setBindGroup(0,n),h.dispatchWorkgroups(o,u,1),h.end(),this.device.queue.submit([f.finish()]),(e.requiresGradient||r.requiresGradient)&&(s.requiresGradient=!0,s.parents=[e,r],s.gradFn=new R([e,r],this.kr)),s}static matmulWGSL=`
    	// C[M,N] = A[M,K] * B[K,N]  (row-major flat buffers)

		struct Params {
		  M : u32,
		  N : u32,
		  K : u32,
		  _pad : u32,
		};
		
		@group(0) @binding(0) var<storage, read> A : array<f32>;
		@group(0) @binding(1) var<storage, read> B : array<f32>;
		@group(0) @binding(2) var<storage, read_write> C : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;
		
		const TILE : u32 = 16u;
		
		var<workgroup> tileA : array<f32, 16u * 16u>;
		var<workgroup> tileB : array<f32, 16u * 16u>;
		
		@compute @workgroup_size(16, 16, 1)
		fn main(
		  @builtin(global_invocation_id) gid : vec3<u32>,
		  @builtin(local_invocation_id)  lid : vec3<u32>,
		) {
		  let row : u32 = gid.y;
		  let col : u32 = gid.x;
		
		  let inBounds : bool = (row < params.M) && (col < params.N);
		
		  let lidx : u32 = lid.y * TILE + lid.x;
		
		  var acc : f32 = 0.0;
		  let numTiles : u32 = (params.K + TILE - 1u) / TILE;
		
		  for (var t : u32 = 0u; t < numTiles; t = t + 1u) {
			let kBase : u32 = t * TILE;
		
			// Load A tile element (or 0)
			let aCol : u32 = kBase + lid.x;
			if ((row < params.M) && (aCol < params.K)) {
			  tileA[lidx] = A[row * params.K + aCol];
			} else {
			  tileA[lidx] = 0.0;
			}
		
			// Load B tile element (or 0)
			let bRow : u32 = kBase + lid.y;
			if ((bRow < params.K) && (col < params.N)) {
			  tileB[lidx] = B[bRow * params.N + col];
			} else {
			  tileB[lidx] = 0.0;
			}
		
			workgroupBarrier();
		
			for (var i : u32 = 0u; i < TILE; i = i + 1u) {
			  acc = acc + tileA[lid.y * TILE + i] * tileB[i * TILE + lid.x];
			}
		
			workgroupBarrier();
		  }
		
		  // Only write valid output elements
		  if (inBounds) {
			C[row * params.N + col] = acc;
		  }
		}
	`}class z{constructor(e,r){this.savedTensors=e,this.kr=r}name="MatAddBackward";backward(e){return[e,e]}}class b extends p{constructor(e,r,s){super(e,b.matAddWGSL,s),this.device=e,this.tm=r,this.kr=s,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(2);paramsBuf;run(e,r,s){if(e.shape.length!==2||r.shape.length!==2||s!==void 0&&s.shape.length!==2)throw new Error("MatAdd: expected 2D tensors");if(e.shape[0]!==r.shape[0]||e.shape[1]!==r.shape[1])throw new Error("MatAdd: tensor shapes must match");const a=e.shape[0],t=e.shape[1];this.params[0]=a,this.params[1]=t,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),s=s??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[a,t]);const i=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:s.buffer}},{binding:3,resource:{buffer:this.paramsBuf}}]}),n=d(t,16),o=d(a,16),u=this.device.createCommandEncoder(),f=u.beginComputePass();return f.setPipeline(this.pipeline),f.setBindGroup(0,i),f.dispatchWorkgroups(n,o,1),f.end(),this.device.queue.submit([u.finish()]),(e.requiresGradient||r.requiresGradient)&&(s.requiresGradient=!0,s.parents=[e,r],s.gradFn=new z([e,r],this.kr)),s}static matAddWGSL=`
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> A : array<f32>;
		@group(0) @binding(1) var<storage, read> B : array<f32>;
		@group(0) @binding(2) var<storage, read_write> C : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(
		  @builtin(global_invocation_id) gid : vec3<u32>,
		) {
		  let row : u32 = gid.y;
		  let col : u32 = gid.x;

		  if (row < params.M && col < params.N) {
		    let idx : u32 = row * params.N + col;
		    C[idx] = A[idx] + B[idx];
		  }
		}
	`}class I{constructor(e,r){this.savedTensors=e,this.kr=r}name="BiasAddBackward";backward(e){const[r,s]=this.savedTensors,a=r.requiresGradient?e:null,t=s.requiresGradient?this.kr.sumReduce.run(e):null;return[a,t]}}class w extends p{constructor(e,r,s){super(e,w.biasAddWGSL,s),this.device=e,this.tm=r,this.kr=s,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(2);paramsBuf;run(e,r,s){if(e.shape.length!==2)throw new Error("BiasAdd: input must be 2D tensor");const a=r.shape.length===1?r.shape[0]:r.shape.length===2&&r.shape[0]===1?r.shape[1]:-1;if(a===-1)throw new Error("BiasAdd: bias must be [N] or [1, N]");if(e.shape[1]!==a)throw new Error(`BiasAdd: input columns (${e.shape[1]}) must match bias size (${a})`);const t=e.shape[0],i=e.shape[1];this.params[0]=t,this.params[1]=i,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),s=s??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[t,i]);const n=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:s.buffer}},{binding:3,resource:{buffer:this.paramsBuf}}]}),o=d(i,16),u=d(t,16),f=this.device.createCommandEncoder(),h=f.beginComputePass();return h.setPipeline(this.pipeline),h.setBindGroup(0,n),h.dispatchWorkgroups(o,u,1),h.end(),this.device.queue.submit([f.finish()]),(e.requiresGradient||r.requiresGradient)&&(s.requiresGradient=!0,s.parents=[e,r],s.gradFn=new I([e,r],this.kr)),s}static biasAddWGSL=`
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> bias : array<f32>;
		@group(0) @binding(2) var<storage, read_write> output : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(
		  @builtin(global_invocation_id) gid : vec3<u32>,
		) {
		  let row : u32 = gid.y;
		  let col : u32 = gid.x;

		  if (row < params.M && col < params.N) {
		    let idx : u32 = row * params.N + col;
		    output[idx] = input[idx] + bias[col];
		  }
		}
	`}class F{constructor(e,r){this.savedTensors=e,this.kr=r}name="ReLUBackward";backward(e){const[r]=this.savedTensors;return[this.kr.reluBackward.run(e,r)]}}class B extends p{constructor(e,r,s){super(e,B.reluWGSL,s),this.device=e,this.tm=r,this.kr=s,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}static RELU_OUTPUT="relu_out";params=new Uint32Array(2);paramsBuf;run(e,r){if(e.shape.length!==2||r!==void 0&&r.shape.length!==2)throw new Error("RELU: expected 2D tensor");const s=e.shape[0],a=e.shape[1];this.params[0]=s,this.params[1]=a,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),r=r??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[s,a]);const t=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:this.paramsBuf}}]}),i=d(a,16),n=d(s,16),o=this.device.createCommandEncoder(),u=o.beginComputePass();return u.setPipeline(this.pipeline),u.setBindGroup(0,t),u.dispatchWorkgroups(i,n,1),u.end(),this.device.queue.submit([o.finish()]),e.requiresGradient&&(r.requiresGradient=!0,r.parents=[e],r.gradFn=new F([e],this.kr)),r}static reluWGSL=`
    	// RELU: remove negative values from a tensor.

		struct Params {
		  M : u32,
		  N : u32,
		};
		
		@group(0) @binding(0) var<storage, read> A : array<f32>;
		@group(0) @binding(1) var<storage, read_write> B : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;
		
		@compute @workgroup_size(16, 16, 1)
		fn main(
		  @builtin(global_invocation_id) gid : vec3<u32>,
		) {
		  let row : u32 = gid.y;
		  let col : u32 = gid.x;
		
		  let inBounds : bool = (row < params.M) && (col < params.N);

		  // Only write valid output elements
		  if (inBounds) {
		  	let f: f32 = A[row * params.N + col];
			B[row * params.N + col] = max(0f, f);
		  }
		}
	`}class Y{constructor(e,r){this.savedTensors=e,this.kr=r}name="SoftmaxBackward";backward(e){const r=this.savedTensors[1];return[this.kr.softmaxBackward.run(e,r)]}}class U extends p{constructor(e,r,s){super(e,U.softmaxWGSL,s),this.device=e,this.tm=r,this.kr=s,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(2);paramsBuf;run(e,r){if(e.shape.length!==2)throw new Error("Softmax: expected 2D tensor");const s=e.shape[0],a=e.shape[1];this.params[0]=s,this.params[1]=a,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),r=r??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[s,a]);const t=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:this.paramsBuf}}]}),i=this.device.createCommandEncoder(),n=i.beginComputePass();return n.setPipeline(this.pipeline),n.setBindGroup(0,t),n.dispatchWorkgroups(d(s,256),1,1),n.end(),this.device.queue.submit([i.finish()]),e.requiresGradient&&(r.requiresGradient=!0,r.parents=[e],r.gradFn=new Y([e,r],this.kr)),r}static softmaxWGSL=`
		// Softmax: exp(x_i - max) / sum(exp(x_j - max))
		// Applied per row for numerical stability.

		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> A : array<f32>;
		@group(0) @binding(1) var<storage, read_write> B : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;

		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let row = gid.x;
			if (row >= params.M) {
				return;
			}

			let N = params.N;
			let base = row * N;

			// Find max for numerical stability
			var maxVal = A[base];
			for (var i = 1u; i < N; i = i + 1u) {
				maxVal = max(maxVal, A[base + i]);
			}

			// Compute exp(x - max) and sum
			var sum = 0.0;
			for (var i = 0u; i < N; i = i + 1u) {
				let e = exp(A[base + i] - maxVal);
				B[base + i] = e;
				sum = sum + e;
			}

			// Normalize
			let invSum = 1.0 / sum;
			for (var i = 0u; i < N; i = i + 1u) {
				B[base + i] = B[base + i] * invSum;
			}
		}
	`}class q{constructor(e,r){this.savedTensors=e,this.kr=r}name="SoftmaxCrossEntropyBackward";backward(e){const[r,s]=this.savedTensors;return[this.kr.softmaxCEBackward.run(r,s)]}}class G extends p{constructor(e,r,s){super(e,G.xentropyWGSL,s),this.device=e,this.tm=r,this.kr=s,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(2);paramsBuf;run(e,r,s){if(e.shape.length!==2)throw new Error("CrossEntropy: logits must be 2D tensor");if(r.shape.length!==2)throw new Error("CrossEntropy: labels must be 2D tensor");if(e.shape[0]!==r.shape[0]||e.shape[1]!==r.shape[1])throw new Error("CrossEntropy: logits and labels must have same shape");const a=e.shape[0],t=e.shape[1];this.params[0]=a,this.params[1]=t,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),s=s??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[a,1]);const i=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:s.buffer}},{binding:3,resource:{buffer:this.paramsBuf}}]}),n=this.device.createCommandEncoder(),o=n.beginComputePass();return o.setPipeline(this.pipeline),o.setBindGroup(0,i),o.dispatchWorkgroups(d(a,256),1,1),o.end(),this.device.queue.submit([n.finish()]),e.requiresGradient&&(s.requiresGradient=!0,s.parents=[e],s.gradFn=new q([e,r],this.kr)),s}static xentropyWGSL=`
		struct Params {
		  M : u32,
		  N : u32,
		};
		
		@group(0) @binding(0) var<storage, read> logits : array<f32>;
		@group(0) @binding(1) var<storage, read> labels : array<f32>; // one-hot
		@group(0) @binding(2) var<storage, read_write> loss : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;
		
		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
		  let row = gid.x;
		  if (row >= params.M) { return; }
		
		  let N = params.N;
		  let base = row * N;
		
		  // 1) max logit for stability
		  var m = logits[base];
		  for (var i = 1u; i < N; i = i + 1u) {
			let z = logits[base + i];
			if (z > m) { m = z; }
		  }
		
		  // 2) logsumexp
		  var sumExp = 0.0;
		  for (var i = 0u; i < N; i = i + 1u) {
			sumExp = sumExp + exp(logits[base + i] - m);
		  }
		  let logSumExp = log(sumExp) + m;
		
		  // 3) cross entropy: -sum y_i * (z_i - logsumexp)
		  var ce = 0.0;
		  for (var i = 0u; i < N; i = i + 1u) {
			let y = labels[base + i];
			let z = logits[base + i];
			ce = ce + y * (logSumExp - z);
		  }
		
		  loss[row] = ce;
		}

	`}class W{constructor(e,r){this.savedTensors=e,this.kr=r}name="DropoutBackward";backward(e){const[r]=this.savedTensors;return[this.kr.elemMul.run(e,r)]}}class P extends p{constructor(e,r,s){super(e,P.dropoutWGSL,s),this.device=e,this.tm=r,this.kr=s,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(2);paramsBuf;run(e,r,s){if(e.shape.length!==2)throw new Error("Dropout: input must be 2D tensor");if(r.shape.length!==2)throw new Error("Dropout: mask must be 2D tensor");if(e.shape[0]!==r.shape[0]||e.shape[1]!==r.shape[1])throw new Error("Dropout: input and mask shapes must match");const a=e.shape[0],t=e.shape[1];this.params[0]=a,this.params[1]=t,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),s=s??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[a,t]);const i=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:s.buffer}},{binding:3,resource:{buffer:this.paramsBuf}}]}),n=d(t,16),o=d(a,16),u=this.device.createCommandEncoder(),f=u.beginComputePass();return f.setPipeline(this.pipeline),f.setBindGroup(0,i),f.dispatchWorkgroups(n,o,1),f.end(),this.device.queue.submit([u.finish()]),e.requiresGradient&&(s.requiresGradient=!0,s.parents=[e],s.gradFn=new W([r],this.kr)),s}static dropoutWGSL=`
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> mask : array<f32>;
		@group(0) @binding(2) var<storage, read_write> output : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(
		  @builtin(global_invocation_id) gid : vec3<u32>,
		) {
		  let row : u32 = gid.y;
		  let col : u32 = gid.x;

		  if (row < params.M && col < params.N) {
		    let idx : u32 = row * params.N + col;
		    output[idx] = input[idx] * mask[idx];
		  }
		}
	`}class y extends p{constructor(e,r,s){super(e,y.transposeWGSL,s),this.device=e,this.tm=r,this.kr=s,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(2);paramsBuf;run(e,r){if(e.shape.length!==2)throw new Error("Transpose: input must be 2D tensor");const s=e.shape[0],a=e.shape[1];if(this.params[0]=s,this.params[1]=a,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),r=r??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[a,s]),r.shape[0]!==a||r.shape[1]!==s)throw new Error("Transpose: output shape must be [N, M]");const t=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:this.paramsBuf}}]}),i=d(a,16),n=d(s,16),o=this.device.createCommandEncoder(),u=o.beginComputePass();return u.setPipeline(this.pipeline),u.setBindGroup(0,t),u.dispatchWorkgroups(i,n,1),u.end(),this.device.queue.submit([o.finish()]),r}static transposeWGSL=`
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> output : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let col : u32 = gid.x;  // output column = input row
			let row : u32 = gid.y;  // output row = input column

			if (row >= params.M || col >= params.N) {
				return;
			}

			// input[row, col] -> output[col, row]
			let inIdx = row * params.N + col;
			let outIdx = col * params.M + row;
			output[outIdx] = input[inIdx];
		}
	`}class S extends p{constructor(e,r,s){super(e,S.sumReduceWGSL,s),this.device=e,this.tm=r,this.kr=s,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(2);paramsBuf;run(e,r){if(e.shape.length!==2)throw new Error("SumReduce: input must be 2D tensor");const s=e.shape[0],a=e.shape[1];this.params[0]=s,this.params[1]=a,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),r=r??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[1,a]);const t=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:this.paramsBuf}}]}),i=this.device.createCommandEncoder(),n=i.beginComputePass();return n.setPipeline(this.pipeline),n.setBindGroup(0,t),n.dispatchWorkgroups(d(a,256),1,1),n.end(),this.device.queue.submit([i.finish()]),r}static sumReduceWGSL=`
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> output : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;

		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let col = gid.x;
			if (col >= params.N) {
				return;
			}

			var sum = 0.0;
			for (var row = 0u; row < params.M; row = row + 1u) {
				sum = sum + input[row * params.N + col];
			}

			output[col] = sum;
		}
	`}class v extends p{constructor(e,r,s){super(e,v.elemMulWGSL,s),this.device=e,this.tm=r,this.kr=s,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(2);paramsBuf;run(e,r,s){if(e.shape.length!==2||r.shape.length!==2)throw new Error("ElementwiseMul: inputs must be 2D tensors");if(e.shape[0]!==r.shape[0]||e.shape[1]!==r.shape[1])throw new Error("ElementwiseMul: input shapes must match");const a=e.shape[0],t=e.shape[1];this.params[0]=a,this.params[1]=t,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),s=s??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[a,t]);const i=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:s.buffer}},{binding:3,resource:{buffer:this.paramsBuf}}]}),n=d(t,16),o=d(a,16),u=this.device.createCommandEncoder(),f=u.beginComputePass();return f.setPipeline(this.pipeline),f.setBindGroup(0,i),f.dispatchWorkgroups(n,o,1),f.end(),this.device.queue.submit([u.finish()]),s}static elemMulWGSL=`
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> A : array<f32>;
		@group(0) @binding(1) var<storage, read> B : array<f32>;
		@group(0) @binding(2) var<storage, read_write> C : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let col = gid.x;
			let row = gid.y;

			if (row >= params.M || col >= params.N) {
				return;
			}

			let idx = row * params.N + col;
			C[idx] = A[idx] * B[idx];
		}
	`}class x extends p{constructor(e,r,s){super(e,x.reluBackwardWGSL,s),this.device=e,this.tm=r,this.kr=s,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(2);paramsBuf;run(e,r,s){if(e.shape.length!==2||r.shape.length!==2)throw new Error("ReLUBackward: inputs must be 2D tensors");if(e.shape[0]!==r.shape[0]||e.shape[1]!==r.shape[1])throw new Error("ReLUBackward: shapes must match");const a=e.shape[0],t=e.shape[1];this.params[0]=a,this.params[1]=t,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),s=s??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[a,t]);const i=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:s.buffer}},{binding:3,resource:{buffer:this.paramsBuf}}]}),n=d(t,16),o=d(a,16),u=this.device.createCommandEncoder(),f=u.beginComputePass();return f.setPipeline(this.pipeline),f.setBindGroup(0,i),f.dispatchWorkgroups(n,o,1),f.end(),this.device.queue.submit([u.finish()]),s}static reluBackwardWGSL=`
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> gradOutput : array<f32>;
		@group(0) @binding(1) var<storage, read> savedInput : array<f32>;
		@group(0) @binding(2) var<storage, read_write> gradInput : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let col = gid.x;
			let row = gid.y;

			if (row >= params.M || col >= params.N) {
				return;
			}

			let idx = row * params.N + col;
			// gradient flows through only where input > 0
			let mask = select(0.0, 1.0, savedInput[idx] > 0.0);
			gradInput[idx] = gradOutput[idx] * mask;
		}
	`}class C extends p{constructor(e,r,s){super(e,C.softmaxCEBackwardWGSL,s),this.device=e,this.tm=r,this.kr=s,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(2);paramsBuf;run(e,r,s){if(e.shape.length!==2||r.shape.length!==2)throw new Error("SoftmaxCEBackward: inputs must be 2D tensors");if(e.shape[0]!==r.shape[0]||e.shape[1]!==r.shape[1])throw new Error("SoftmaxCEBackward: shapes must match");const a=e.shape[0],t=e.shape[1];this.params[0]=a,this.params[1]=t,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),s=s??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[a,t]);const i=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:s.buffer}},{binding:3,resource:{buffer:this.paramsBuf}}]}),n=this.device.createCommandEncoder(),o=n.beginComputePass();return o.setPipeline(this.pipeline),o.setBindGroup(0,i),o.dispatchWorkgroups(d(a,256),1,1),o.end(),this.device.queue.submit([n.finish()]),s}static softmaxCEBackwardWGSL=`
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> logits : array<f32>;
		@group(0) @binding(1) var<storage, read> labels : array<f32>;
		@group(0) @binding(2) var<storage, read_write> gradLogits : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let row = gid.x;
			if (row >= params.M) {
				return;
			}

			let N = params.N;
			let base = row * N;

			// 1) Find max for numerical stability
			var maxVal = logits[base];
			for (var i = 1u; i < N; i = i + 1u) {
				maxVal = max(maxVal, logits[base + i]);
			}

			// 2) Compute exp(x - max) and sum
			var sum = 0.0;
			for (var i = 0u; i < N; i = i + 1u) {
				let e = exp(logits[base + i] - maxVal);
				gradLogits[base + i] = e;
				sum = sum + e;
			}

			// 3) Normalize to get softmax, then subtract labels
			let invSum = 1.0 / sum;
			for (var i = 0u; i < N; i = i + 1u) {
				let softmax_i = gradLogits[base + i] * invSum;
				gradLogits[base + i] = softmax_i - labels[base + i];
			}
		}
	`}class k extends p{constructor(e,r,s){super(e,k.softmaxBackwardWGSL,s),this.device=e,this.tm=r,this.kr=s,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(2);paramsBuf;run(e,r,s){if(e.shape.length!==2||r.shape.length!==2)throw new Error("SoftmaxBackward: inputs must be 2D tensors");if(e.shape[0]!==r.shape[0]||e.shape[1]!==r.shape[1])throw new Error("SoftmaxBackward: shapes must match");const a=e.shape[0],t=e.shape[1];this.params[0]=a,this.params[1]=t,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),s=s??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[a,t]);const i=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:s.buffer}},{binding:3,resource:{buffer:this.paramsBuf}}]}),n=this.device.createCommandEncoder(),o=n.beginComputePass();return o.setPipeline(this.pipeline),o.setBindGroup(0,i),o.dispatchWorkgroups(d(a,256),1,1),o.end(),this.device.queue.submit([n.finish()]),s}static softmaxBackwardWGSL=`
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> gradOutput : array<f32>;
		@group(0) @binding(1) var<storage, read> softmaxOut : array<f32>;
		@group(0) @binding(2) var<storage, read_write> gradInput : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let row = gid.x;
			if (row >= params.M) {
				return;
			}

			let N = params.N;
			let base = row * N;

			// Compute dot = sum(dS * S) for this row
			var dot = 0.0;
			for (var i = 0u; i < N; i = i + 1u) {
				dot = dot + gradOutput[base + i] * softmaxOut[base + i];
			}

			// Compute gradInput = S * (dS - dot)
			for (var i = 0u; i < N; i = i + 1u) {
				let idx = base + i;
				gradInput[idx] = softmaxOut[idx] * (gradOutput[idx] - dot);
			}
		}
	`}class T extends p{constructor(e,r,s){super(e,T.scalarMulWGSL,s),this.device=e,this.tm=r,this.paramsBuf=e.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Float32Array(4);paramsBuf;run(e,r,s){if(e.shape.length!==2)throw new Error("ScalarMul: input must be 2D tensor");const a=e.shape[0],t=e.shape[1],i=new Uint32Array(this.params.buffer);i[0]=a,i[1]=t,this.params[2]=r,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),s=s??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[a,t]);const n=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:s.buffer}},{binding:2,resource:{buffer:this.paramsBuf}}]}),o=d(t,16),u=d(a,16),f=this.device.createCommandEncoder(),h=f.beginComputePass();return h.setPipeline(this.pipeline),h.setBindGroup(0,n),h.dispatchWorkgroups(o,u,1),h.end(),this.device.queue.submit([f.finish()]),s}static scalarMulWGSL=`
		struct Params {
		  M : u32,
		  N : u32,
		  scalar : f32,
		  _pad : u32,
		};

		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> output : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let col = gid.x;
			let row = gid.y;

			if (row >= params.M || col >= params.N) {
				return;
			}

			let idx = row * params.N + col;
			output[idx] = input[idx] * params.scalar;
		}
	`}class _ extends p{constructor(e,r,s){super(e,_.inplaceAddWGSL,s),this.device=e,this.tm=r,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(2);paramsBuf;run(e,r){if(e.shape.length!==2||r.shape.length!==2)throw new Error("InplaceAdd: inputs must be 2D tensors");if(e.shape[0]!==r.shape[0]||e.shape[1]!==r.shape[1])throw new Error("InplaceAdd: tensor shapes must match");const s=e.shape[0],a=e.shape[1];this.params[0]=s,this.params[1]=a,this.device.queue.writeBuffer(this.paramsBuf,0,this.params);const t=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:this.paramsBuf}}]}),i=d(a,16),n=d(s,16),o=this.device.createCommandEncoder(),u=o.beginComputePass();u.setPipeline(this.pipeline),u.setBindGroup(0,t),u.dispatchWorkgroups(i,n,1),u.end(),this.device.queue.submit([o.finish()])}static inplaceAddWGSL=`
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read_write> targetTensor : array<f32>;
		@group(0) @binding(1) var<storage, read> source : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let col = gid.x;
			let row = gid.y;

			if (row >= params.M || col >= params.N) {
				return;
			}

			let idx = row * params.N + col;
			targetTensor[idx] = targetTensor[idx] + source[idx];
		}
	`}class D extends p{constructor(e,r,s){super(e,D.sumAllWGSL,s),this.device=e,this.tm=r,this.paramsBuf=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}params=new Uint32Array(2);paramsBuf;run(e,r){if(e.shape.length!==2)throw new Error("SumAll: input must be 2D tensor");const s=e.shape[0],a=e.shape[1];s*a>65536&&console.warn("SumAll: tensor size > 65536, consider using hierarchical reduction"),this.params[0]=s,this.params[1]=a,this.device.queue.writeBuffer(this.paramsBuf,0,this.params),r=r??this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[1,1]);const i=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e.buffer}},{binding:1,resource:{buffer:r.buffer}},{binding:2,resource:{buffer:this.paramsBuf}}]}),n=this.device.createCommandEncoder(),o=n.beginComputePass();return o.setPipeline(this.pipeline),o.setBindGroup(0,i),o.dispatchWorkgroups(1,1,1),o.end(),this.device.queue.submit([n.finish()]),r}static sumAllWGSL=`
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> output : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;

		var<workgroup> sharedData : array<f32, 256>;

		@compute @workgroup_size(256, 1, 1)
		fn main(
			@builtin(local_invocation_id) lid : vec3<u32>,
		) {
			let totalSize = params.M * params.N;
			let tid = lid.x;

			// Each thread sums a strided portion of the input
			var sum = 0.0;
			var idx = tid;
			while (idx < totalSize) {
				sum = sum + input[idx];
				idx = idx + 256u;
			}

			sharedData[tid] = sum;
			workgroupBarrier();

			// Parallel reduction in sharedData memory
			if (tid < 128u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 128u]; }
			workgroupBarrier();
			if (tid < 64u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 64u]; }
			workgroupBarrier();
			if (tid < 32u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 32u]; }
			workgroupBarrier();
			if (tid < 16u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 16u]; }
			workgroupBarrier();
			if (tid < 8u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 8u]; }
			workgroupBarrier();
			if (tid < 4u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 4u]; }
			workgroupBarrier();
			if (tid < 2u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 2u]; }
			workgroupBarrier();
			if (tid < 1u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 1u]; }
			workgroupBarrier();

			// Thread 0 writes the result
			if (tid == 0u) {
				output[0] = sharedData[0];
			}
		}
	`}class X{matmul;matadd;biasadd;relu;softmax;crossEntropy;dropout;transpose;sumReduce;elemMul;reluBackward;softmaxBackward;softmaxCEBackward;scalarMul;inplaceAdd;sumAll;constructor(e,r){this.matmul=new l(e,r,this),this.matadd=new b(e,r,this),this.biasadd=new w(e,r,this),this.relu=new B(e,r,this),this.softmax=new U(e,r,this),this.crossEntropy=new G(e,r,this),this.dropout=new P(e,r,this),this.transpose=new y(e,r,this),this.sumReduce=new S(e,r,this),this.elemMul=new v(e,r,this),this.reluBackward=new x(e,r,this),this.softmaxBackward=new k(e,r,this),this.softmaxCEBackward=new C(e,r,this),this.scalarMul=new T(e,r,this),this.inplaceAdd=new _(e,r,this),this.sumAll=new D(e,r,this)}}class L{constructor(e,r,s,a,t=!1){this.name=e,this.buffer=r,this.usage=s,this.shape=a,this.size=a.reduce((i,n)=>i*n,1),this.requiresGradient=t}size;gradient;gradFn=void 0;parents=void 0;requiresGradient;sizeInBytes(){return this.size*4}backward(){if(!this.gradient)throw new Error("Tensor has no gradient")}zeroGrad(e){this.gradient&&e.zeros(this.gradient)}}function E(c,e){return Math.ceil(c/e)*e}class V{constructor(e){this.device=e}tensors=new Map;readback=null;pendingDestroy=[];scopeName="";scopeCounter=0;beginScope(e){this.scopeName=e,this.scopeCounter=0}getScopedTensor(e,r,s){const a=`_${this.scopeName}_${this.scopeCounter++}`;return this.getTensorBuffer(a,e,r,s)}getTensorBuffer(e,r,s,a=void 0){const t=this.tensors.get(e),i=s.reduce((u,f)=>u*f,1)*4;if(t&&t.sizeInBytes()>=i&&t.usage===r){if(a&&this.writeBufferF32(t.buffer,a),t.shape.length===s.length&&t.shape.every((h,O)=>h===s[O]))return t;const f=new L(e,t.buffer,r,s);return this.tensors.set(e,f),f}const n=this.device.createBuffer({size:E(i,256),usage:r});t&&this.pendingDestroy.push(t.buffer);const o=new L(e,n,r,s);return this.tensors.set(e,o),a&&this.writeBufferF32(n,a),o}writeBufferF32(e,r,s=0){this.device.queue.writeBuffer(e,s,r.buffer,r.byteOffset,r.byteLength)}async readBuffer(e,r,s=0){const a=this.ensureReadback(r),t=this.device.createCommandEncoder();t.copyBufferToBuffer(e,s,a,0,r),this.device.queue.submit([t.finish()]),await a.mapAsync(GPUMapMode.READ,0,r);const n=a.getMappedRange(0,r).slice(0);return a.unmap(),new Float32Array(n)}ensureReadback(e){if(e=E(e,256),this.readback!==null&&this.readback.size>=e)return this.readback.buffer;this.readback&&(this.pendingDestroy.push(this.readback.buffer),this.readback=null);const r=this.device.createBuffer({size:e,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});return this.readback={buffer:r,size:e},r}ones(e,r="ones"){const s=this.getTensorBuffer(r,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,e);return this.writeBufferF32(s.buffer,new Float32Array(e.reduce((a,t)=>a*t,1)).fill(1)),s}scopedOnes(e){const r=this.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,e);return this.writeBufferF32(r.buffer,new Float32Array(e.reduce((s,a)=>s*a,1)).fill(1)),r}scopedZeros(e){const r=this.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,e);return this.writeBufferF32(r.buffer,new Float32Array(e.reduce((s,a)=>s*a,1)).fill(0)),r}zeros(e){this.writeBufferF32(e.buffer,new Float32Array(e.size).fill(0))}async flushDestroyQueue(){if(this.pendingDestroy.length!==0){await this.device.queue.onSubmittedWorkDone();for(const e of this.pendingDestroy)try{e.destroy()}catch(r){console.error("Failed to destroy tensor buffer "+r)}this.pendingDestroy.length=0}}async destroyAll(){await this.device.queue.onSubmittedWorkDone();for(const{buffer:e}of this.tensors.values())try{e.destroy()}catch(r){console.error("Failed to destroy tensor buffer "+r)}if(this.tensors.clear(),this.readback){try{this.readback.buffer.destroy()}catch(e){console.error("Failed to destroy readback buffer "+e)}this.readback=null}for(const e of this.pendingDestroy)try{e.destroy()}catch(r){console.error("Failed to destroy scheduled to destroy tensor buffer "+r)}this.pendingDestroy.length=0}}class M{constructor(e,r,s,a,t,i){this.batchSize=e,this.imageData=r,this.oneImageSize=s,this.labelsData=a,this.oneLabelSize=t,this.maxSize=i,this.workingImageBuffer=new Float32Array(s*this.batchSize),this.workingLabelBuffer=new Float32Array(t*this.batchSize);const n=Math.min(this.maxSize,this.imageData.length/s),o=Math.min(this.maxSize,this.labelsData.length/t);if(n!==o)throw new Error("MNIST data source: image and label data have different sizes");this.iteratorSize=o,this.restart()}currentIndex=0;iteratorSize=0;datasetIndices=new Uint32Array(0);workingImageBuffer;workingLabelBuffer;getCurrentIndex(){return this.currentIndex}getSize(){return this.iteratorSize}hasNext(){return this.currentIndex<this.iteratorSize}next(){const e=Math.min(this.batchSize,this.iteratorSize-this.currentIndex);for(let r=0;r<e;r++){const s=this.datasetIndices[this.currentIndex++],a=s*this.oneImageSize;this.workingImageBuffer.set(this.imageData.subarray(a,a+this.oneImageSize),r*this.oneImageSize);const t=s*this.oneLabelSize;this.workingLabelBuffer.set(this.labelsData.subarray(t,t+this.oneLabelSize),r*this.oneLabelSize)}return{data:e<this.batchSize?this.workingImageBuffer.subarray(0,e*this.oneImageSize):this.workingImageBuffer,labels:e<this.batchSize?this.workingLabelBuffer.subarray(0,e*this.oneLabelSize):this.workingLabelBuffer,size:e}}restart(){this.currentIndex=0,this.datasetIndices=M.getRandomDataset(this.imageData.length/this.oneImageSize)}static getRandomDataset(e){const r=new Uint32Array(e);for(let s=0;s<r.length;s++)r[s]=s;for(let s=e-1;s>0;s--){const a=Math.floor(Math.random()*(s+1));[r[s],r[a]]=[r[a],r[s]]}return r}}class g{trainData=null;trainLabelsData=null;testData=null;testLabelsData=null;static imageSize=784;testImagesCount=1e4;trainImagesCount=6e4;maxTrainSize=1/0;maxTestSize=1/0;constructor(){}toFloat32(e){const r=new Float32Array(e.length);for(let s=0;s<e.length;s++)r[s]=e[s]/255;return r}getIterator(e,r,s,a){return new M(e,r,g.imageSize,s,10,a)}getTrainIterator(e){if(!this.trainData||!this.trainLabelsData)throw new Error("MNISTDatasource: train data not loaded. Call load() first.");return this.getIterator(e,this.trainData,this.trainLabelsData,this.maxTrainSize)}getTestIterator(e){if(!this.testData||!this.testLabelsData)throw new Error("MNISTDatasource: test data not loaded. Call load() first.");return this.getIterator(e,this.testData,this.testLabelsData,this.maxTestSize)}onehot(e){const r=new Float32Array(e.length*10);for(let s=0;s<e.length;s++){const a=e[s];r[s*10+a]=1}return r}async load(e){const r=await fetch(`${e}/train-images.idx3-ubyte`);this.trainData=this.toFloat32(new Uint8Array(await r.arrayBuffer(),16));const s=await fetch(`${e}/train-labels.idx1-ubyte`);this.trainLabelsData=this.onehot(new Uint8Array(await s.arrayBuffer(),8));const a=await fetch(`${e}/t10k-images.idx3-ubyte`);this.testData=this.toFloat32(new Uint8Array(await a.arrayBuffer(),16));const t=await fetch(`${e}/t10k-labels.idx1-ubyte`);this.testLabelsData=this.onehot(new Uint8Array(await t.arrayBuffer(),8)),this.testImagesCount=this.testData.length/g.imageSize,this.trainImagesCount=this.trainData.length/g.imageSize}}class ${inputTensor;layers=[];constructor(...e){for(const r of e)this.layers.push(r)}forward(e,r){this.inputTensor=e;for(const s of this.layers)e=s.forward(e,r);return e}parameters(){return this.layers.flatMap(e=>e.parameters())}zeroGrad(e){for(const r of this.parameters())r.gradient&&e.writeBufferF32(r.gradient.buffer,new Float32Array(r.size).fill(0))}}class A{constructor(e,r,s){this.tm=e,this.kr=r;const a=[s.inputFeatures,s.outputFeatures];this.name=s.name,this.weights=e.getTensorBuffer(`${s.name}_weights`,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,a,s.initializer(a,s.inputFeatures)),this.weights.requiresGradient=!0,s.useBias&&(this.bias=e.getTensorBuffer(`${s.name}_bias`,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[1,s.outputFeatures]),this.bias.requiresGradient=!0)}inputTensor;name;weights;bias;forward(e,r){this.inputTensor=e;const s=this.tm.getTensorBuffer(`${this.name}_mmout`,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[e.shape[0],this.weights.shape[1]]),a=this.kr.matmul.run(e,this.weights,s);if(!this.bias)return a;const t=this.tm.getTensorBuffer(`${this.name}_sumout`,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,[e.shape[0],this.weights.shape[1]]);return this.kr.biasadd.run(s,this.bias,t)}parameters(){const e=[this.weights];return this.bias!==void 0&&e.push(this.bias),e}}function m(c,e,r){const s=Math.sqrt(6/e),a=c.reduce((t,i)=>t*i,1);r=r??new Float32Array(a);for(let t=0;t<a;t++)r[t]=(Math.random()*2-1)*s;return r}class K{constructor(e,r,s="ReLU"){this.tm=e,this.kr=r,this.name=s}inputTensor;forward(e,r){this.inputTensor=e;const s=this.tm.getScopedTensor(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,e.shape);return this.kr.relu.run(e,s)}parameters(){return[]}}class j{constructor(e,r,s=m){this.tm=e,this.initializer=s,this.firstLayer=new A(e,r,{name:"first",inputFeatures:g.imageSize,outputFeatures:128,useBias:!0,initializer:s}),this.secondLayer=new A(e,r,{name:"second",inputFeatures:128,outputFeatures:10,useBias:!0,initializer:s}),this.model=new $(this.firstLayer,new K(e,r,"ReLU1"),this.secondLayer)}model;firstLayer;secondLayer;async readSnapshot(){const e="data/trained_768_128_10",r=await fetch(`${e}/model-first_weights.bin`),s=new Float32Array(await r.arrayBuffer()),a=await fetch(`${e}/model-first_bias.bin`),t=new Float32Array(await a.arrayBuffer());this.tm.writeBufferF32(this.firstLayer.parameters()[0].buffer,s),this.tm.writeBufferF32(this.firstLayer.parameters()[1].buffer,t);const i=await fetch(`${e}/model-second_weights.bin`),n=new Float32Array(await i.arrayBuffer()),o=await fetch(`${e}/model-second_bias.bin`),u=new Float32Array(await o.arrayBuffer());return this.tm.writeBufferF32(this.secondLayer.parameters()[0].buffer,n),this.tm.writeBufferF32(this.secondLayer.parameters()[1].buffer,u),[s,t,n,u]}async restart(){this.model.zeroGrad(this.tm),this.tm.zeros(this.firstLayer.parameters()[1]),this.tm.zeros(this.secondLayer.parameters()[1]);const e=this.firstLayer.parameters()[0].shape;this.tm.writeBufferF32(this.firstLayer.parameters()[0].buffer,m(e,e[0]));const r=this.secondLayer.parameters()[0].shape;this.tm.writeBufferF32(this.secondLayer.parameters()[0].buffer,m(r,r[0]))}}export{N as G,X as K,j as M,V as T,g as a};
