import type { RoleAccess } from "./user";

/*
 * Which SSO the scheduler signs in through, matching the ODB the backend reads
 * (the backend only uses the GPP development and production environments).
 * Resolved from the hostname like the lucuma-ts apps; `VITE_SSO_URL` overrides
 * it for any other deployment.
 */

const SSO_URIS = {
  development: "https://sso-dev.gpp.gemini.edu",
  production: "https://sso.gpp.gemini.edu",
} as const;

const PRODUCTION_HOSTS = new Set(["schedule.gemini.edu"]);

export function ssoUriFor(hostname: string): string {
  return PRODUCTION_HOSTS.has(hostname)
    ? SSO_URIS.production
    : SSO_URIS.development;
}

export const SSO_URI: string =
  import.meta.env.VITE_SSO_URL || ssoUriFor(window.location.hostname);

/** The lowest role that may use the scheduler. The gate is a UI courtesy; the
 *  backend must verify the bearer token to actually enforce it. */
export const REQUIRED_ACCESS: RoleAccess = "staff";
