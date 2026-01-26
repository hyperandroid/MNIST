import { defineConfig } from 'vite'
import { resolve } from 'path'
import { viteStaticCopy } from 'vite-plugin-static-copy'

export default defineConfig({
  base: '/MNIST/',
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: 'data',
          dest: '.',
        },
      ],
    }),
  ],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        guess: resolve(__dirname, 'guess.html'),
        train_and_test: resolve(__dirname, 'train_and_test.html'),
      },
    },
  },
})
