import { useEffect, type ReactNode } from "react";
import { FaLock } from "react-icons/fa";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import orcidLogo from "@/assets/orcid-logo.svg";
import { REQUIRED_ACCESS } from "./environment";
import { signIn, signOut, startSession } from "./session";
import { useAuth } from "./store";
import { accessAtLeast, currentAccess, displayName } from "./user";

/**
 * Starts the session keeper and gates the app: nothing below mounts until the
 * session check settles, so no subscription ever opens signed out. Signed out
 * → the ORCID sign-in screen; signed in below staff → access denied; staff or
 * better → the app.
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const { status, user } = useAuth();

  useEffect(() => startSession(), []);

  switch (status) {
    case "checking":
      return (
        <GateScreen>
          <p className="text-sm opacity-70">Checking your session…</p>
        </GateScreen>
      );
    case "signed-out":
      return (
        <GateScreen>
          <h1 className="text-lg font-semibold">Sign in to the Scheduler</h1>
          <p className="text-sm opacity-70">
            The scheduler is restricted to Gemini staff.
          </p>
          <Button variant="outline" size="lg" onClick={signIn}>
            <img src={orcidLogo} alt="" aria-hidden="true" className="size-5" />
            Login with ORCID
          </Button>
        </GateScreen>
      );
    case "signed-in":
      if (!accessAtLeast(user, REQUIRED_ACCESS)) {
        return (
          <GateScreen>
            <FaLock className="size-8 opacity-70" aria-hidden="true" />
            <h1 className="text-lg font-semibold">Access denied</h1>
            <p className="text-sm opacity-70 max-w-sm">
              {user ? displayName(user) : "You"} signed in as{" "}
              <strong>{currentAccess(user).toUpperCase()}</strong>; the
              scheduler needs a <strong>{REQUIRED_ACCESS.toUpperCase()}</strong>{" "}
              role or higher.
            </p>
            <Button variant="outline" onClick={() => void signOut()}>
              Sign out
            </Button>
          </GateScreen>
        );
      }
      return <>{children}</>;
  }
}

function GateScreen({ children }: { children: ReactNode }) {
  return (
    <main
      className={cn(
        "dark:bg-black bg-white dark:text-white text-black",
        "flex flex-col items-center justify-center gap-4 h-screen text-center px-4",
      )}
    >
      <span className="text-2xl font-bold tracking-wide">Schedule</span>
      {children}
    </main>
  );
}
