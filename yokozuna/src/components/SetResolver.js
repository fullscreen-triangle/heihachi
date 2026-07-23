'use client'
/**
 * SetResolver — renders a resolved continuous item as its signature.
 *
 * Shows the ordered tracklist (the paper's signature): matched positions carry
 * a name (source 'meta' or 'acoustic'), residual positions are shown honestly
 * as "🔍 identifying…" while the acoustic worker runs, or "ID (unidentified)"
 * when even acoustics could not place them (Rem. 6.11 — honest residual, not a
 * forced guess). A banner reports the best set-index match (Thm. 5.4: the
 * sequence, not the content, identifies the set).
 */

function fmtTime(secs) {
    if (secs == null || isNaN(secs)) return ''
    const h = Math.floor(secs / 3600)
    const m = Math.floor((secs % 3600) / 60)
    const s = Math.floor(secs % 60)
    const mm = h ? String(m).padStart(2, '0') : String(m)
    return `${h ? h + ':' : ''}${mm}:${String(s).padStart(2, '0')}`
}

function Row({ item, resolving }) {
    const isResidual = !item.name
    const identifying = isResidual && resolving && item.source !== 'id'

    return (
        <div className="flex items-center gap-3 px-4 py-2.5 border-b border-white/5 last:border-0">
            <span className="text-white/30 text-xs font-mono w-12 text-right shrink-0">
                {fmtTime(item.tStart)}
            </span>
            <div className="flex-1 min-w-0">
                {item.name ? (
                    <div className="text-sm text-white/90 truncate">{item.name}</div>
                ) : identifying ? (
                    <div className="text-sm text-amber-300/70 italic">🔍 identifying…</div>
                ) : (
                    <div className="text-sm text-white/40 italic">ID (unidentified)</div>
                )}
            </div>
            {item.source && (
                <span
                    className={
                        'text-[10px] px-1.5 py-0.5 rounded shrink-0 ' +
                        (item.source === 'meta'
                            ? 'bg-emerald-500/15 text-emerald-300/80'
                            : item.source === 'acoustic'
                              ? 'bg-violet-500/20 text-violet-300/90'
                              : 'bg-white/5 text-white/40')
                    }
                >
                    {item.source === 'meta'
                        ? 'name'
                        : item.source === 'acoustic'
                          ? 'acoustic'
                          : 'ID'}
                </span>
            )}
        </div>
    )
}

export default function SetResolver({ resolved, resolving, match, onClose, onPlay }) {
    if (!resolved && !resolving) return null

    const sig = resolved?.signature || []
    const residualCount = (resolved?.residual || []).length
    const named = sig.filter((s) => s.name).length
    const total = sig.length

    return (
        <div className="max-w-2xl mx-auto mt-2 bg-black/90 backdrop-blur-lg border border-white/10 rounded-xl overflow-hidden">
            {/* Header */}
            <div className="flex items-start gap-3 px-4 py-3 border-b border-white/10">
                {resolved?.thumbnail && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                        src={resolved.thumbnail}
                        alt=""
                        className="w-14 h-14 rounded object-cover shrink-0"
                    />
                )}
                <div className="flex-1 min-w-0">
                    <div className="text-white text-sm font-semibold truncate">
                        {resolved?.title || 'Resolving…'}
                    </div>
                    <div className="text-white/40 text-xs truncate">
                        {resolved?.author}
                        {resolved?.platform && (
                            <span className="ml-2 uppercase text-white/25">{resolved.platform}</span>
                        )}
                    </div>
                    {total > 0 && (
                        <div className="text-white/30 text-[11px] mt-1">
                            {named}/{total} placed
                            {residualCount > 0 && ` · ${residualCount} residual`}
                            {resolved?.acousticUsed && ' · acoustic descent used'}
                        </div>
                    )}
                </div>
                {onClose && (
                    <button
                        onClick={onClose}
                        className="text-white/30 hover:text-white/70 text-lg leading-none shrink-0"
                        aria-label="Close"
                    >
                        ×
                    </button>
                )}
            </div>

            {/* Set-index match banner */}
            {match && match.record && (
                <div className="px-4 py-2 bg-violet-500/10 border-b border-violet-500/20">
                    <p className="text-violet-200/80 text-xs">
                        {match.admissible ? 'Resembles a set you resolved' : 'Closest prior set'}:{' '}
                        <span className="font-medium">{match.record.title}</span>{' '}
                        <span className="text-violet-300/60">
                            (align score {match.score.toFixed(2)} · {match.accounted.length}/
                            {sig.length} aligned)
                        </span>
                    </p>
                </div>
            )}

            {/* Worker status */}
            {resolved?.workerError && (
                <div className="px-4 py-2 bg-amber-500/10 border-b border-amber-500/20">
                    <p className="text-amber-200/70 text-[11px]">
                        Acoustic worker unavailable — showing metadata-only signature. (
                        {resolved.workerError})
                    </p>
                </div>
            )}

            {/* Signature / tracklist */}
            {resolving && sig.length === 0 ? (
                <div className="px-4 py-6 text-center text-white/40 text-xs">
                    Acquiring signature…
                </div>
            ) : (
                <div className="max-h-80 overflow-y-auto">
                    {sig.map((item) => (
                        <Row key={item.pos} item={item} resolving={resolving} />
                    ))}
                    {sig.length === 0 && !resolving && (
                        <div className="px-4 py-6 text-center text-white/30 text-xs">
                            No tracklist found in the link’s metadata.
                            {resolved?.workerConfigured
                                ? ' Acoustic segmentation did not recover one either.'
                                : ' Configure the acoustic worker to resolve it from audio.'}
                        </div>
                    )}
                </div>
            )}

            {/* Actions */}
            {onPlay && resolved?.platform && (
                <div className="px-4 py-3 border-t border-white/10">
                    <button
                        onClick={() => onPlay(resolved)}
                        className="w-full py-2 rounded-md bg-white/10 border border-white/20 text-white/70
                            text-xs font-medium hover:bg-white/20 transition-all"
                    >
                        Open source ↗
                    </button>
                </div>
            )}
        </div>
    )
}
