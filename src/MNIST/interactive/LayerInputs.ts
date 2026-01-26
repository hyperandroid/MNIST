/**
 * render layers buffers.
 */

export class LayerInputs {

	private ctx: CanvasRenderingContext2D;

	constructor() {
		const canvas = document.createElement("canvas");
		canvas.width = 2 * 784 + 20;
		canvas.height = 65*4 + 20;
		this.ctx = canvas.getContext("2d")!;
		this.ctx.font = "20px Arial";
		this.ctx.fillStyle = "#333";
		this.ctx.fillRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);

		const parent = document.getElementById("layers") ?? document.body;
		parent.appendChild(canvas);
	}

	render(layersData: Float32Array[], layers: string[]) {
		this.ctx.fillStyle = "#333";
		this.ctx.fillRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
		layersData.forEach((d, i) => {
			this.drawLayer(d, 10, 10 + i * 65, layers[i], i > 0);
		});
	}

	private drawLayer(data: Float32Array, x: number, y: number, label: string, outline = false) {

		let max = -Infinity;
		let min = Infinity;
		for(let i =0; i<data.length; i++) {
			if (data[i] > max) max = data[i];
			if (data[i] < min) min = data[i];
		}
		for(let i =0; i<data.length; i++) {
			data[i] = (data[i] - min) / (max - min);
		}

		const scalew = 784/data.length * 2;

		for (let i = 0; i < data.length; i++) {
			const v = data[i];
			const c = Math.floor(v*255);
			this.ctx.fillStyle = `rgb(${c},${c},${c*.8})`;
			this.ctx.fillRect(x + i * scalew, y+30, scalew, 20);

			if (outline) {
				this.ctx.strokeStyle = "white";
				this.ctx.strokeRect(x + i * scalew, y+30, scalew, 20);
			}
		}
		this.ctx.fillStyle = "White";
		this.ctx.fillText(label, x, y + 20);
	}
}
