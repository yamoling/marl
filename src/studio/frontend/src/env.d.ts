/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "1" enables the mock API at build/dev time. */
  readonly VITE_MOCK?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
