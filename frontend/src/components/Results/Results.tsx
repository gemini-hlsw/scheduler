import { cn } from "@/lib/utils";
import { NightPlanType } from "../../types";
import NightPlan from "./NightPlan";
import {
  Carousel,
  CarouselApi,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel";
import { useState } from "react";

export default function Results({ plans }: { plans: NightPlanType[] }) {
  if (!plans || plans.length === 0) {
    return null;
  }
  const [currentSlide, setCurrentSlide] = useState(0);
  const [api, setApi] = useState<CarouselApi | null>(null);

  function goToSlide(idx: number) {
    return () => {
      api?.scrollTo(idx);
      setCurrentSlide(idx);
    };
  }

  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-2 p-3 flex-wrap",
        "dark:bg-white/20 bg-black/10"
      )}
    >
      <h1 className="font-bold w-full">Results</h1>
      <Carousel
        className="w-full px-4"
        opts={{
          loop: true,
        }}
        setApi={setApi}
      >
        <CarouselContent>
          {plans.map((plan) => (
            <CarouselItem key={plan.nightIndex}>
              <NightPlan nightPlan={plan} />
            </CarouselItem>
          ))}
        </CarouselContent>
        <CarouselPrevious
          className="absolute -left-2 h-20 w-5 rounded-md"
          onClick={goToSlide(
            currentSlide - 1 < 0 ? plans.length - 1 : currentSlide - 1
          )}
        />
        <CarouselNext
          className="absolute -right-2.5 h-20 w-6 rounded-md"
          onClick={goToSlide(
            currentSlide + 1 > plans.length - 1 ? 0 : currentSlide + 1
          )}
        />
      </Carousel>
      <div>
        {api && (
          <div className="flex flex-wrap justify-center gap-2">
            {plans.map((plan, idx) => {
              const isSelected = idx === currentSlide;
              const mornTwi = plan.timeEntriesBySite[0].mornTwilight;
              const timelineDate =
                mornTwi.substring(0, mornTwi.indexOf("T")) ?? "";
              return (
                <button
                  key={`plan${plan.nightIndex}`}
                  className={cn(
                    "h-4 rounded-full text-xs font-mono font-bold",
                    "overflow-hidden",
                    isSelected
                      ? "bg-blue-500 dark:bg-blue-500 w-24"
                      : "bg-gray-400 dark:bg-gray-600 w-4",
                    "hover:bg-blue-500 dark:hover:bg-blue-500",
                    "hover:w-24",
                    "group",
                    "transition-all"
                  )}
                  onClick={goToSlide(idx)}
                >
                  <p
                    className={cn(
                      !isSelected && "group-hover:block hidden",
                      "text-white"
                    )}
                  >
                    {timelineDate}
                  </p>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
