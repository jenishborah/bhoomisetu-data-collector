export default function PageHeader({ eyebrow, title, children, updatedAt }) {
  return <header className="page-heading"><div>{eyebrow && <span className="section-label">{eyebrow}</span>}<h1>{title}</h1>{children && <div className="page-description">{children}</div>}</div>{updatedAt && <div className="data-status"><span>LAST UPDATED</span><strong>{updatedAt}</strong></div>}</header>;
}
