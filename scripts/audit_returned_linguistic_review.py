from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import openpyxl


def text(value):
    return str(value).strip() if value is not None else ""


def predict_suffix(token, rules):
    for rule in sorted(rules, key=lambda r: (r['priority'], -len(r['suffix']), r['suffix'])):
        if token in rule['exceptions'] or not token.endswith(rule['suffix']):
            continue
        candidate = token[:-len(rule['suffix'])] + rule['replacement']
        letters = sum(unicodedata.category(c)[0] in {'L', 'M'} for c in candidate)
        if letters >= rule['minimum_stem_letter_count']:
            return candidate
    return token


def audit(path, root):
    workbook = openpyxl.load_workbook(path, data_only=False)
    original = json.loads((root / 'results/corpus/linguistic-review-data.json').read_text(encoding='utf-8'))
    assignment = json.loads((root / 'configs/linguistic_review.json').read_text(encoding='utf-8'))
    issues = []
    reviewer = text(workbook['Instructions']['B3'].value)
    completion = workbook['Instructions']['B7'].value
    completion_date = completion.date().isoformat() if isinstance(completion, datetime) else completion.isoformat() if isinstance(completion, date) else text(completion)
    try:
        date.fromisoformat(completion_date)
    except ValueError:
        issues.append('Review completion date is absent or invalid.')
    if reviewer != assignment['reviewer']['name']:
        issues.append(f"Reviewer name differs. Assignment recorded {assignment['reviewer']['name']}. Returned workbook recorded {reviewer}.")
    stopwords = {}
    for language, sheet_name in [('en', 'Stopwords EN'), ('lus', 'Stopwords Mizo')]:
        sheet = workbook[sheet_name]
        entries = []
        expected_tokens = {entry['token'] for entry in original['stopwords'][language]['entries']}
        for row in range(6, 206):
            token = text(sheet.cell(row, 2).value)
            decision = text(sheet.cell(row, 6).value)
            if decision not in {'retain', 'exclude_from_index'}:
                issues.append(f'{sheet_name}!F{row} has an invalid or missing decision.')
            entries.append({'token': token, 'decision': decision, 'reviewer_basis': text(sheet.cell(row, 7).value), 'source_cell': f'{sheet_name}!F{row}'})
        if len({entry['token'] for entry in entries}) != len(entries) or {entry['token'] for entry in entries} != expected_tokens:
            issues.append(f'{sheet_name} token identities differ from the issued workbook.')
        stopwords[language] = {'counts': dict(Counter(e['decision'] for e in entries)), 'entries': entries}
    rule_decisions = []
    rules = []
    sheet = workbook['Mizo Rules']
    for row in range(6, 46):
        decision = text(sheet.cell(row, 6).value)
        item = {'candidate': text(sheet.cell(row, 1).value), 'decision': decision, 'reviewer_basis': text(sheet.cell(row, 12).value), 'row': row}
        rule_decisions.append(item)
        if decision not in {'accept', 'reject', 'modify'}:
            issues.append(f'Mizo Rules!F{row} has an invalid or missing decision.')
        if decision not in {'accept', 'modify'}:
            continue
        values = [sheet.cell(row, col).value for col in range(7, 13)]
        if any(value is None or text(value) == '' for value in values):
            issues.append(f'Mizo Rules row {row} has missing rule fields.')
            continue
        suffix, replacement, minimum, priority, exceptions, basis = values
        rules.append({'suffix': text(suffix), 'replacement': '' if text(replacement) == '<empty>' else text(replacement), 'minimum_stem_letter_count': int(minimum), 'priority': int(priority), 'exceptions': [] if text(exceptions) == '<none>' else [x.strip() for x in text(exceptions).split(',') if x.strip()], 'basis': text(basis)})
    if {item['candidate'] for item in rule_decisions} != {item['suffix'] for item in original['mizo_suffix_candidates']}:
        issues.append('Mizo suffix candidate identities differ from the issued workbook.')
    annotations = {}
    for language, sheet_name, start, end in [('en', 'English Stems', 9, 48), ('lus', 'Mizo Stems', 6, 55)]:
        sheet = workbook[sheet_name]
        items = []
        for row in range(start, end + 1):
            token = text(sheet.cell(row, 1).value)
            if language == 'en':
                decision = text(sheet.cell(row, 6).value)
                if decision not in {'accept_suggestion', 'correct'}:
                    issues.append(f'English Stems!F{row} has an invalid or missing decision.')
                expected = text(sheet.cell(row, 3 if decision == 'accept_suggestion' else 7).value)
                predicted = text(sheet.cell(row, 3).value)
                note = text(sheet.cell(row, 9).value)
            else:
                expected = text(sheet.cell(row, 5).value)
                predicted = predict_suffix(token, rules)
                note = text(sheet.cell(row, 6).value)
            if not expected:
                issues.append(f'{sheet_name} row {row} has no expected stem.')
            items.append({'token': token, 'expected_stem': expected, 'predicted_stem': predicted, 'correct': predicted == expected, 'reviewer_note': note, 'row': row})
        if {item['token'] for item in items} != {item['token'] for item in original['annotations'][language]}:
            issues.append(f'{sheet_name} token identities differ from the issued workbook.')
        annotations[language] = {'item_count': len(items), 'correct_count': sum(item['correct'] for item in items), 'items': items}
        annotations[language]['accuracy'] = annotations[language]['correct_count'] / len(items)
    algorithm_decision = text(workbook['English Stems']['B4'].value)
    if algorithm_decision != 'approve':
        issues.append('English algorithm approval is absent or requires revision.')
    return {'source_path': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'reviewer': reviewer, 'assigned_reviewer': assignment['reviewer']['name'], 'completion_date': completion_date, 'english_algorithm_decision': algorithm_decision, 'issues': issues, 'stopwords': stopwords, 'mizo_rule_counts': dict(Counter(item['decision'] for item in rule_decisions)), 'mizo_rule_decisions': rule_decisions, 'mizo_rules': rules, 'annotations': annotations}


def report(result):
    lines = ['# Linguistic review', '', f"Returned reviewer: {result['reviewer']}", '', f"Assigned reviewer: {result['assigned_reviewer']}", '', f"Completion date: {result['completion_date']}", '', '| Review item | Result |', '|---|---|']
    for language, label in [('en', 'English'), ('lus', 'Mizo')]:
        counts = result['stopwords'][language]['counts']
        lines.append(f"| {label} stopwords | {counts.get('exclude_from_index', 0)} excluded, {counts.get('retain', 0)} retained |")
        annotations = result['annotations'][language]
        lines.append(f"| {label} annotation agreement | {annotations['correct_count']} of {annotations['item_count']} ({annotations['accuracy']:.1%}) |")
    counts = result['mizo_rule_counts']
    lines.append(f"| Mizo suffix candidates | {counts.get('accept', 0)} accepted, {counts.get('reject', 0)} rejected |")
    lines.extend(['', 'English agreement used the issued Snowball suggestions. Mizo agreement used the current single-pass suffix engine with the returned rules. These annotation results were diagnostic and were not independent held-out accuracy estimates.', '', '## Review issues', ''])
    lines.extend(f'- {issue}' for issue in result['issues'])
    if not result['issues']:
        lines.append('Reviewer identity and all required workbook fields were validated.')
    lines.extend(['', 'The accepted in rule requires a lexicon check in its review basis. The current engine has no lexicon check. The rejected ahte candidate describes sequential application of te and ah. The current engine returns after one rule. These implementation conditions require resolution before production stemming.', '', '## Annotation disagreements', '', '| Language | Token | Current output | Reviewed stem | Workbook row |', '|---|---|---|---|---|'])
    for language, label in [('en', 'English'), ('lus', 'Mizo')]:
        for item in result['annotations'][language]['items']:
            if not item['correct']:
                lines.append(f"| {label} | {item['token']} | {item['predicted_stem']} | {item['expected_stem']} | {item['row']} |")
    lines.extend(['', 'The returned workbook was preserved. The audit did not modify production review resources.', '', 'Every count and comparison was generated by scripts/audit_returned_linguistic_review.py.', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workbook', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = audit(args.workbook, root)
    (root / 'results/corpus/returned-linguistic-review-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (root / 'docs/returned-linguistic-review.md').write_text(report(result), encoding='utf-8')
    print(report(result))


if __name__ == '__main__':
    main()
