import { Badge } from "@/components/ui/badge";
export function ObsClassBadge({ obsClass }: { obsClass: string }) {
  function getSeverity(oc: string) {
    switch (oc) {
      case "SCIENCE":
        return "text-white bg-green-500";
      case "PROGCAL":
        return "text-white bg-yellow-500";
      case "PARTNERCAL":
        return "text-white bg-red-500";
      case "ACQ":
        return "text-white bg-blue-500";
      case "ACQCAL":
        return "text-white bg-purple-500";
      case "DAYCAL":
        return "text-white bg-gray-500";
      default:
        return "";
    }
  }

  return <Badge className={getSeverity(obsClass)}>{obsClass}</Badge>;
}
