const labels = {
  "3a": "Preliminary notification",
  "3A": "Acquisition processing",
  "3D": "Declaration / acquisition",
};

function StageFlow({ counts = {}, currentStage, compact = false }) {
  const stages = ["3a", "3A", "3D"];
  return <div className={`stage-flow ${compact ? "compact" : ""}`}>
    {stages.map((stage, index) => <div className="stage-flow-part" key={stage}>
      <div className={`stage-node ${currentStage === stage ? "current" : ""}`}><strong>{counts[stage] ?? stage}</strong><span>{stage}</span></div>
      <div className="stage-copy"><strong>{stage}</strong><small>{labels[stage]}</small></div>
      {index < stages.length - 1 && <span className="stage-arrow" aria-hidden="true">→</span>}
    </div>)}
  </div>;
}

export default StageFlow;
