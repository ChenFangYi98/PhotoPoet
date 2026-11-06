# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Apply the explicitly recorded assistant pilot review; never mark it human QA."""
import hashlib
import json
import re
from photopoet_build import ROOT, database, jd, han, validate_text, export

def drop_samples(db, asset):
    db.execute('DELETE FROM ngrams WHERE sample IN (SELECT id FROM samples WHERE asset=?)', (asset,))
    db.execute('DELETE FROM samples WHERE asset=?', (asset,))

def main():
    db = database()
    review = json.loads((ROOT / '05_review/assistant_pilot_edits.json').read_text(encoding='utf-8'))
    for asset, reason in review['reject_assets'].items():
        drop_samples(db, asset)
        db.execute("UPDATE assets SET state='rejected',reason=? WHERE id=?", ('assistant_visual_review:' + reason, asset))
    # Apply newer strict language/format rules to already generated pilot rows.
    for row in db.execute('SELECT * FROM samples').fetchall():
        if row['stage'] == 'pretrain' and re.search('[A-Za-z]', row['target']):
            drop_samples(db, row['asset'])
            db.execute("UPDATE assets SET state='pending',facts=NULL,reason='caption_english_requeued' WHERE id=?", (row['asset'],))
        elif row['stage'] == 'sft' and validate_text(row['task'], row['target']):
            db.execute('DELETE FROM ngrams WHERE sample=?', (row['id'],))
            db.execute('DELETE FROM samples WHERE id=?', (row['id'],))
    for edit in review['edits']:
        row = db.execute('SELECT * FROM samples WHERE asset=? AND task=?', (edit['asset'], edit['task'])).fetchone()
        if not row: continue
        assert validate_text(edit['task'], edit['target']) is None, edit
        qc = json.loads(row['review'])
        qc['post_edit'] = {'editor': review['reviewer'], 'human_review': False, 'reason': edit['reason']}
        qc['status'] = 'assistant_reviewed_pending_human'
        digest = hashlib.sha256(re.sub(r'\s+', '', edit['target']).encode()).hexdigest()
        db.execute('UPDATE samples SET target=?,target_sha=?,review=? WHERE id=?', (edit['target'], digest, jd(qc), row['id']))
        db.execute('DELETE FROM ngrams WHERE sample=?', (row['id'],))
        chars = han(edit['target'])
        db.executemany('INSERT OR IGNORE INTO ngrams VALUES(?,?,?)', [('sft', chars[i:i+8], row['id']) for i in range(max(0, len(chars)-7))])
    db.commit()
    export(db)

if __name__ == '__main__':
    main()
