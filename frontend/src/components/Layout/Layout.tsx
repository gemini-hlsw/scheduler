import { ReactNode } from "react";
import Header from "./Header/Header";
import Navbar from "./Navbar/Navbar";
import { cn } from "../../lib/utils";

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div
      className={cn(
        "dark:bg-black bg-white dark:text-white text-black",
        "flex flex-col h-screen overflow-y-auto w-full"
      )}
    >
      <Header title="schedule" />
      <div className={"flex flex-col md:flex-row w-full grow"}>
        <Navbar />
        <main className="p-3 order-1 md:order-1 w-full md:w-[calc(100%-24px)]">
          {children}
        </main>
      </div>
    </div>
  );
}
