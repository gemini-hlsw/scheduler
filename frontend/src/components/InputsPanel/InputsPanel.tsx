import UploadButton from "./UploadButton/UploadButton";
import { cn } from "@/lib/utils";

export default function InputsPanel({
  vertical = false,
}: {
  vertical?: boolean;
}) {
  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-2 p-3 flex-wrap",
        "dark:bg-white/20 bg-black/10"
      )}
    >
      <h1 className="font-bold">Sources</h1>
      <div
        className={cn(
          "flex gap-2",
          vertical ? "flex-col" : "flex-row flex-wrap"
        )}
      >
        <UploadButton label="Faults"></UploadButton>
        <UploadButton label="GMOS Conf"></UploadButton>
        <UploadButton label="rToOs"></UploadButton>
        <UploadButton label="Weather"></UploadButton>
        <UploadButton label="Calendar"></UploadButton>
      </div>
    </div>
  );
}
