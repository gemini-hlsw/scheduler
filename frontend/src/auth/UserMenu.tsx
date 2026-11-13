import { FaSignOutAlt } from "react-icons/fa";
import { Button } from "@/components/ui/button";
import { signOut } from "./session";
import { useAuth } from "./store";
import { currentAccess, displayName } from "./user";

/** The signed-in user's name and role, with a sign-out button, for the header. */
export function UserMenu() {
  const { user } = useAuth();
  if (!user) return null;

  return (
    <div className="flex flex-row items-center gap-2 text-xs">
      <span className="whitespace-nowrap">
        {displayName(user)}{" "}
        <span className="opacity-60">({currentAccess(user)})</span>
      </span>
      <Button
        variant="outline"
        size="xs"
        onClick={() => void signOut()}
        title="Sign out"
      >
        <FaSignOutAlt />
        <span className="label">Sign out</span>
      </Button>
    </div>
  );
}
