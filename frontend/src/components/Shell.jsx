import { NavLink, Outlet } from "react-router-dom";

const navigation = [
  ["/", "Dashboard", true],
  ["/projects", "Projects"],
  ["/gis", "GIS Map"],
  ["/stage-sentinel", "Stage Sentinel"],
  ["/reports", "Reports"],
];

function Shell() {
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <header>
        <div className="gov-strip">
          <div><strong>GOVERNMENT OF INDIA</strong><span>Land Acquisition Monitoring &amp; Decision Support</span></div>
          <div className="gov-utility">Prototype Information System · Accessibility</div>
        </div>
        <div className="brand-strip">
          <div className="brand-lockup">
            <div className="brand-mark" aria-hidden="true"><span>B</span><i /></div>
            <div>
              <h1>BhoomiSetu</h1>
              <p>From Land Data to Construction-Ready Decisions</p>
            </div>
          </div>
          <div className="system-status"><span className="status-dot" /> <span>System Status</span><strong>Operational</strong></div>
        </div>
        <nav className="main-nav" aria-label="Primary navigation">
          {navigation.map(([path, label, end]) => (
            <NavLink key={path} to={path} end={end} className={({ isActive }) => (isActive ? "active" : "")}>
              {label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main id="main-content" className="main-content"><Outlet /></main>
      <footer className="site-footer">
        <strong>BhoomiSetu</strong><span>Land Acquisition Intelligence &amp; Decision Support Platform</span><span>·</span><span>Synthetic-data prototype</span>
      </footer>
    </div>
  );
}

export default Shell;
