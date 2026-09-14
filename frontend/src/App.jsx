import { BrowserRouter, Route, Routes } from "react-router-dom";

import "./App.css";

import Shell from "./components/Shell";
import Dashboard from "./pages/Dashboard";
import GISMap from "./pages/GISMap";
import ProjectDetail from "./pages/ProjectDetail";
import Projects from "./pages/Projects";
import Reports from "./pages/Reports";
import StageSentinelPage from "./pages/StageSentinelPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route index element={<Dashboard />} />
          <Route path="projects" element={<Projects />} />
          <Route path="projects/:projectId" element={<ProjectDetail />} />
          <Route path="gis" element={<GISMap />} />
          <Route path="stage-sentinel" element={<StageSentinelPage />} />
          <Route path="reports" element={<Reports />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
