import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { ApolloProvider } from "@apollo/client";
import { client } from "./apollo-client";
import { TooltipProvider } from "@/components/ui/tooltip";
import ThemeProvider from "./theme/ThemeProvider";
import App from "./App";
import Home from "./components/Home";
import Validation from "./components/Validation/Validation";
import GlobalStateProvider from "./components/GlobalState/GlobalState";
import Operation from "./components/Operation/Operation";
import { Simulation } from "./components/Simulation/Simulation";
import "./global.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ThemeProvider>
      <TooltipProvider>
        <GlobalStateProvider>
          <ApolloProvider client={client}>
            <BrowserRouter>
              <Routes>
                <Route path="/" element={<App />}>
                  <Route index element={<Home />} />
                  <Route path="operation" element={<Operation />} />
                  <Route path="validation" element={<Validation />} />
                  <Route path="simulation" element={<Simulation />} />
                  <Route path="link1" element={<h1>link1</h1>} />
                  <Route path="link2" element={<h1>link2</h1>} />
                  <Route path="link3" element={<h1>link3</h1>} />
                  <Route path="link4" element={<h1>link4</h1>} />
                  <Route path="link5" element={<h1>link5</h1>} />
                </Route>
              </Routes>
            </BrowserRouter>
          </ApolloProvider>
        </GlobalStateProvider>
      </TooltipProvider>
    </ThemeProvider>
  </React.StrictMode>
);
