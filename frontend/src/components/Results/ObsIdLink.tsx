import { FaExternalLinkAlt } from "react-icons/fa";

// The observation id, linked to its page in Explore when the backend sent one
// (ODB programs only: validation runs read the OCS, which Explore does not know).
export function ObsIdLink({
  obsId,
  exploreUrl,
}: {
  obsId: string;
  exploreUrl?: string | null;
}) {
  if (!exploreUrl) return <>{obsId}</>;

  return (
    <a
      href={exploreUrl}
      target="_blank"
      rel="noopener noreferrer"
      title="Open in Explore"
      className="inline-flex items-center gap-1 text-blue-700 dark:text-blue-300 hover:underline"
    >
      {obsId}
      <FaExternalLinkAlt className="size-3" />
    </a>
  );
}
