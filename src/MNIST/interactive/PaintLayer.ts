const PX = 20;

export class PaintLayer {

	private readonly root: HTMLElement;
	private readonly canvas: HTMLCanvasElement
	private readonly ctx: CanvasRenderingContext2D;

	private readonly canvasOut: HTMLCanvasElement | undefined;
	private readonly ctxOut: CanvasRenderingContext2D | undefined;

	private readonly model: Float32Array;
	private readonly outmodel: Float32Array;

	private capturing = false;

	constructor(
		private readonly onModelChanged: (model: Float32Array) => void,
		private readonly onModelCleared: () => void,
		readonly rows: number = 24,
		readonly columns: number = 24,
		readonly showPadded = false,
	) {
		this.root = document.createElement("div");
		const row = document.getElementById("row")!;
		row.appendChild(this.root);

		this.canvas = document.createElement("canvas");
		this.canvas.width = this.columns * PX;
		this.canvas.height = this.rows * PX;
		this.ctx = this.canvas.getContext("2d")!;

		this.model = new Float32Array(rows * columns);
		this.outmodel = new Float32Array(28*28);

		this.render(this.ctx, this.rows, this.columns, this.model);

		this.canvas.addEventListener("mousedown", _ => {
			this.capturing = true;
		});

		this.canvas.addEventListener("mouseup", _ => {
			this.capturing = false;
		});

		this.canvas.addEventListener("mousemove", (e) => {
			if (!this.capturing) {
				return;
			}

			this.setModel(e);
		});

		this.canvas.addEventListener("click", (e) => {
			this.setModel(e);
		});

		this.root.appendChild(document.createTextNode(`${this.rows}x${this.columns}, padded to 28x28`));
		this.root.appendChild(document.createElement("br"));

		this.root.appendChild(document.createElement("br"));
		this.root.appendChild(this.canvas);
		this.root.appendChild(document.createElement("br"));

		const clearButton = document.createElement("button");
		clearButton.textContent = "Clear";
		clearButton.addEventListener("click", () => {
			this.model.fill(0);
			this.render(this.ctx, this.rows, this.columns, this.model);
			this.onModelCleared();
		});
		this.root.appendChild(clearButton);

		if (showPadded) {
			this.canvasOut = document.createElement("canvas");
			this.canvasOut.width = 28 * PX;
			this.canvasOut.height = 28 * PX;
			this.ctxOut = this.canvasOut.getContext("2d")!;
			this.root.appendChild(document.createElement("br"));
			this.root.appendChild(this.canvasOut);
			this.render(this.ctxOut, 28, 28, this.outmodel);
		}
	}

	private incColor(x: number, y: number, sign: number = 1) {

		if (y>0) {
			this.increment(x,y-1, .33 * sign);
		}
		if (y<this.rows-1) {
			this.increment(x,y+1, .33 * sign);
		}
		if (x>0) {
			this.increment(x-1,y, .33 * sign);
		}
		if (x<this.columns-1) {
			this.increment(x+1,y, .33 * sign);
		}
		this.increment(x,y,sign);
	}

	private increment(x: number, y: number, factor: number) {
		const v = this.model[y * this.columns + x];
		this.model[y * this.columns + x] = Math.max(0, Math.min(1, v + factor));
	}

	private setModel(e: MouseEvent) {

		const x = Math.floor(e.offsetX / PX);
		const y = Math.floor(e.offsetY / PX);

		if (e.altKey) {
			this.incColor(x,y, -1);
		} else {
			this.incColor(x,y);
		}

		this.render(this.ctx, this.rows, this.columns, this.model);

		this.process();

		if (this.showPadded) {
			this.render(this.ctxOut!, 28, 28, this.outmodel);
		}

		this.onModelChanged(this.outmodel);
	}

	private process() {

		// clear dest
		this.outmodel.fill(0);

		// find bounding box
		let minY = Infinity;
		let maxY = -Infinity;
		let minX = Infinity;
		let maxX = -Infinity;

		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.columns; j++) {
				if (this.model[i * this.columns + j] > 0) {
					if (i < minY) minY = i;
					if (i > maxY) maxY = i;
					if (j < minX) minX = j;
					if (j > maxX) maxX = j;
				}
			}
		}

		// calculate copy offsets
		const xoffset = Math.floor((28-(maxX-minX))/2);
		const yoffset = Math.floor((28-(maxY-minY))/2);

		// copy from model into outmodel
		for (let i = 0; i <= maxY-minY; i++) {
			for (let j = 0; j <= maxX-minX; j++) {
				this.outmodel[(i+yoffset) * 28 + j + xoffset] = this.model[(i+minY) * this.columns + j + minX];
			}
		}
	}

	render(ctx: CanvasRenderingContext2D, rows: number, columns: number, model: Float32Array): void {

		ctx.fillStyle = "#111";
		ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
		ctx.strokeStyle = "white";

		for (let i = 0; i < rows; i++) {
			for (let j = 0; j < columns; j++) {
				const v = model[i * columns + j];
				const color = Math.floor(v*255);
				ctx.fillStyle = `rgba(${color},${color},${color*.7},1.0)`;
				ctx.fillRect(j * PX, i * PX, PX, PX);
				ctx.strokeRect(j * PX, i * PX, PX, PX);
			}
		}
	}
}
