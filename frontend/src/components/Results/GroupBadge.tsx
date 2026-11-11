import { Badge } from "@/components/ui/badge";

// One color per group in the plan, so neighbouring groups are told apart.
const GROUP_COLORS = [
  "text-white bg-sky-600",
  "text-white bg-fuchsia-600",
  "text-white bg-amber-600",
  "text-white bg-teal-600",
  "text-white bg-rose-600",
  "text-white bg-indigo-600",
];

// Order of the groups in the plan, used to pick each group's color.
export function planGroupIds(visits: { groupId?: string | null }[]): string[] {
  const ids: string[] = [];
  for (const { groupId } of visits) {
    if (groupId && !ids.includes(groupId)) ids.push(groupId);
  }
  return ids;
}

// Marks a visit that belongs to a consecutive AND group. Visits showing the
// same name and color have to be done together.
export function GroupBadge({
  groupId,
  groupName,
  groupIds,
}: {
  groupId?: string | null;
  groupName?: string | null;
  groupIds: string[];
}) {
  if (!groupId) return null;
  // Unnamed groups fall back to the id, without the program id the row already shows.
  const label = groupName || groupId.split(":").pop();
  const color = GROUP_COLORS[groupIds.indexOf(groupId) % GROUP_COLORS.length];
  return (
    <Badge className={color} title={groupId}>
      {label}
    </Badge>
  );
}
