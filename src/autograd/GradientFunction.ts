import {Tensor} from "../tensor/Tensor";

export interface GradientFunction {

	name: string;
	savedTensors: Tensor[];

	backward(gradOutput: Tensor): Tensor[];
}
