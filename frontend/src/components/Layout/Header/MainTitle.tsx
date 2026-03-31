import { useNavigate } from "react-router-dom";
import { cn } from "../../../lib/utils";

interface MainTitleProps {
  title: string;
}

export default function MainTitle({ title }: MainTitleProps) {
  const splited_title = title.split("");
  const navigate = useNavigate();

  return (
    <div
      className={cn("cursor-pointer uppercase text-sm flex flex-row gap-1")}
      onClick={() => navigate("/")}
    >
      {splited_title.map((letter, index) => (
        <span
          className={cn("animate-jump")}
          style={{ animationDelay: `${index * 0.1}s` }}
          key={index}
        >
          {letter}
        </span>
      ))}
    </div>
  );
}
