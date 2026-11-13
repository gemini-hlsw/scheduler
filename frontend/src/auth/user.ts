/*
 * The signed-in user, decoded from the SSO JWT. Same model as lucuma-ts
 * (packages/common-ui/src/auth.ts) so the role semantics match the other GPP
 * apps. Role hierarchy (increasing power):
 *   guest < pi < ngo < staff < admin < service
 */

export type RoleAccess = "guest" | "pi" | "ngo" | "staff" | "admin" | "service";

const ACCESS_ORDER: Record<RoleAccess, number> = {
  guest: 0,
  pi: 1,
  ngo: 2,
  staff: 3,
  admin: 4,
  service: 5,
};

export interface StandardRole {
  readonly type: "pi" | "ngo" | "staff" | "admin";
  readonly id: string;
  /** Present only for NGO roles. */
  readonly partner?: string;
}

export interface OrcidProfile {
  readonly orcidId: string;
  readonly profile: {
    readonly givenName?: string;
    readonly familyName?: string;
    readonly creditName?: string;
    readonly primaryEmail?: string | null;
  };
}

export interface StandardUser {
  readonly type: "standard";
  readonly id: string;
  readonly role: StandardRole;
  readonly otherRoles: readonly StandardRole[];
  readonly profile: OrcidProfile;
}

export interface GuestUser {
  readonly type: "guest";
  readonly id: string;
}

export interface ServiceUser {
  readonly type: "service";
  readonly id: string;
  readonly name: string;
}

export type User = StandardUser | GuestUser | ServiceUser;

interface TokenPayload {
  "lucuma-user": User;
  exp: number;
}

/** The user and expiry carried by a JWT, or null when it does not decode. */
export function decodeToken(token: string): { user: User; exp: Date } | null {
  const payload = token.split(".")[1];
  if (!payload) return null;
  try {
    // JWTs are base64url; atob only takes standard base64.
    const b64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    const json = new TextDecoder().decode(
      Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)),
    );
    const decoded = JSON.parse(json) as TokenPayload;
    const user = decoded["lucuma-user"];
    if (!user || typeof decoded.exp !== "number") return null;
    return { user, exp: new Date(decoded.exp * 1000) };
  } catch {
    return null;
  }
}

/** The access level of the user's currently active role. */
export function currentAccess(user: User | null): RoleAccess {
  if (!user) return "guest";
  switch (user.type) {
    case "standard":
      return user.role.type;
    case "service":
      return "service";
    case "guest":
      return "guest";
  }
}

export function accessAtLeast(user: User | null, min: RoleAccess): boolean {
  return ACCESS_ORDER[currentAccess(user)] >= ACCESS_ORDER[min];
}

/** One of the user's other roles that meets `min`, to switch to when the
 *  active role falls short. */
export function eligibleOtherRole(
  user: User | null,
  min: RoleAccess,
): StandardRole | null {
  if (user?.type !== "standard") return null;
  return (
    user.otherRoles.find((r) => ACCESS_ORDER[r.type] >= ACCESS_ORDER[min]) ??
    null
  );
}

/** ORCID credit name, then given + family name, then the bare ORCID iD. */
export function displayName(user: User): string {
  switch (user.type) {
    case "guest":
      return "Guest User";
    case "service":
      return `Service user (${user.name})`;
    case "standard": {
      const p = user.profile.profile;
      if (p.creditName) return p.creditName;
      if (p.givenName && p.familyName) return `${p.givenName} ${p.familyName}`;
      return p.familyName || p.givenName || user.profile.orcidId;
    }
  }
}
