import { Field } from "@/components/ui/field";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";

export function SemesterVisibility({
  semesterVisibility,
  setSemesterVisibility,
}: {
  semesterVisibility: boolean;
  setSemesterVisibility: (checked: boolean) => void;
}) {
  return (
    <Field orientation="horizontal" className="w-fit">
      <Label htmlFor={"semesterVisibility"}>Semester Visibility</Label>
      <Checkbox
        id={"semesterVisibility"}
        name={"semesterVisibility"}
        checked={semesterVisibility}
        disabled={false}
        onCheckedChange={(checked) => {
          setSemesterVisibility(checked as boolean);
        }}
      />
    </Field>
  );
}
