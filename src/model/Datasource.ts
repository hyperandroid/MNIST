
export interface Datasource {
	load(path: string): Promise<void>;
}
