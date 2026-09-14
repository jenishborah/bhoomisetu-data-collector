export default function ErrorState({ error, onRetry, title = "Data service unavailable" }) {
  return <section className="error-state" role="alert"><span className="error-kicker">SERVICE NOTICE</span><h2>{title}</h2><p>{error || "The requested data could not be loaded."}</p>{onRetry && <button className="button primary" onClick={onRetry}>Retry request</button>}</section>;
}
