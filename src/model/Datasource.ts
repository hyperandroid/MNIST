
export interface Datasource {
	load(): Promise<void>;
}
