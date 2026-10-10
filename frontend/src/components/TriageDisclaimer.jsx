/** Shown under every reply about symptoms. The text comes from the server, in the reply's language. */
export default function TriageDisclaimer({ text, lang }) {
  if (!text) return null;
  return (
    <p className="triage-disclaimer" role="note" lang={lang}>
      {text}
    </p>
  );
}
