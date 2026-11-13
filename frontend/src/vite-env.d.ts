/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** SSO origin override; by default chosen from the hostname. */
  readonly VITE_SSO_URL?: string;
}
