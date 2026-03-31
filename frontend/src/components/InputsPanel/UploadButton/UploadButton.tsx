import { Input } from "@/components/ui/input";
import { Field, FieldLabel } from "@/components/ui/field";

interface UploadProps {
  label: string;
}

export default function UploadButton({ label }: UploadProps) {
  return (
    <Field orientation="horizontal" className="w-fit">
      <FieldLabel htmlFor="file-upload" className="w-fit">
        {label}
      </FieldLabel>
      <Input id="file-upload" type="file" className="w-24" />
    </Field>
  );
}
