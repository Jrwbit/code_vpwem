/* Synthetic browser regressions; no real accounts, exports, or fixed user paths.
 * Install playwright-core outside this repository, then run:
 *   PLAYWRIGHT_MODULE=/path/to/playwright-core CHROME_PATH=/path/to/chrome \
 *     node tools/personal_finance/test_dashboard.cjs
 * CHROME_PATH is optional when the standard Chrome channel is installed.
 * FINANCE_NO_SANDBOX=1 is available for isolated CI environments.
 * FINANCE_SCREENSHOT_DIR optionally saves synthetic desktop/mobile previews.
 */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');

const record = (id, kind, amount, fields = {}) => ({
  id, kind, amount, time: '2026-01-10 12:00:00', month: '2026-01',
  direction: ['income', 'refund', 'reimbursement'].includes(kind) ? 'in' : 'out',
  category: '示例原分类', spending_category: '日用购物',
  spending_category_reason: '仅用于自动化测试的合成依据', spending_category_review: true,
  source: 'example-app', sources: ['example-app'], source_label: '示例应用',
  counterparty: '虚构商户', description: '合成商品摘要', review: false,
  entity_ids: ['example-person'], entity_roles: {'example-person': ['counterparty']},
  evidence: [{id: `evidence-${id}`, source_label: '示例应用', amount, currency: 'CNY'}],
  ...fields,
});

const active = [
  record('purchase', 'expense', 10_000, {spending_category: '餐饮',
    evidence: [{id: 'original-purchase-evidence', amount: 10_000, source_label: '示例应用'}],
    entity_ids: ['example-person', 'example-person', 'example-payment'],
    entity_roles: {'example-person': ['payer'], 'example-payment': ['payment']},
    funded_by_other: true}),
  record('partial-refund', 'refund', 2_500, {spending_category: '餐饮',
    related_id: 'original-purchase-evidence', entity_ids: ['example-person', 'example-payment'],
    entity_roles: {'example-person': ['payer'], 'example-payment': ['payment']},
    funded_by_other: true}),
  ...Array.from({length: 119}, (_, index) => {
    const month = index % 3 + 1;
    return record(`row-${String(index).padStart(3, '0')}`, index === 0 ? 'income' : 'expense',
      (index % 19 + 1) * 137, {time: `2026-${String(month).padStart(2, '0')}-${String(index % 25 + 1).padStart(2, '0')} 09:30:00`,
        month: `2026-${String(month).padStart(2, '0')}`,
        counterparty: ['虚构甲', '虚构乙', '虚构丙'][index % 3],
        spending_category: index % 2 ? '日用购物' : '交通',
        description: index === 4 ? '<img src=x onerror="window.syntheticXss=1">' : '合成商品摘要'});
  }),
];
const removed = [
  record('removed-purchase', 'excluded', 9_900, {full_refund_removed: true, full_refund_group: 'group-one', removal_original_kind: 'expense'}),
  record('removed-refund', 'excluded', 9_900, {direction: 'in', full_refund_removed: true, full_refund_group: 'group-one', removal_original_kind: 'refund'}),
  record('only-removed-purchase', 'excluded', 4_500, {entity_ids: ['only-removed'], full_refund_removed: true, full_refund_group: 'group-two', removal_original_kind: 'expense'}),
  record('only-removed-refund', 'excluded', 4_500, {entity_ids: ['only-removed'], direction: 'in', full_refund_removed: true, full_refund_group: 'group-two', removal_original_kind: 'refund'}),
];
const data = {
  meta: {title: '合成回归账本', period_start: '2026-01-01', period_end: '2026-10-22'},
  sources: [{id: 'example-app', label: '示例应用', count: 125}],
  entities: [
    {id: 'example-person', label: '虚构人物', type: 'person', aliases: ['人物别名'], featured: true},
    {id: 'example-payment', label: '示例亲情卡', type: 'payment', aliases: [], featured: true},
    {id: 'only-removed', label: '只有全退的虚构对象', type: 'merchant', aliases: [], featured: true},
  ],
  transactions: [...active, ...removed],
  audit: {raw_count: 125, canonical_count: 125, duplicate_count: 0,
    full_refunds: {summary: {group_count: 2, record_count: 4}}},
};

function expectedOrder(rows, field, direction) {
  return [...rows].sort((a, b) => {
    const value = field === 'amount' ? a.amount - b.amount : String(a[field] || '').localeCompare(String(b[field] || ''), 'zh-CN', {numeric: true, sensitivity: 'base'});
    return value ? value * (direction === 'asc' ? 1 : -1) : a.id.localeCompare(b.id);
  }).map(row => row.id);
}
async function displayedIds(page, scope = 'main') {
  return page.locator(`#${scope === 'main' ? 'transactionRows' : 'entityTransactionRows'} .ledger-row`).evaluateAll(rows => rows.map(row => row.dataset.id));
}
async function downloadText(download) {
  const stream = await download.createReadStream(), chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return Buffer.concat(chunks).toString('utf8');
}

(async () => {
  const browser = await chromium.launch({headless: true,
    ...(process.env.CHROME_PATH ? {executablePath: process.env.CHROME_PATH} : {channel: 'chrome'}),
    args: process.env.FINANCE_NO_SANDBOX === '1' ? ['--no-sandbox'] : []});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    page.setDefaultTimeout(15_000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(pathToFileURL(path.join(__dirname, 'web', 'index.html')).href);
    assert(await page.locator('#emptyState').isVisible());
    const legacy = structuredClone(data);
    for (const row of legacy.transactions) {
      if (row.full_refund_removed) row.kind = row.removal_original_kind;
      delete row.full_refund_removed;
      delete row.full_refund_group;
      delete row.removal_original_kind;
    }
    await page.evaluate(value => FinanceDashboard.loadData(value), legacy);
    const fingerprint = await page.evaluate(() => FinanceDashboard.getFingerprint());
    const previousChanges = {
      'removed-purchase': {kind: 'expense', category: '旧审核分类', review: true},
      'removed-refund': {kind: 'income', category: '旧审核收入', review: false},
      purchase: {kind: 'expense', category: '保留原有审核', review: false},
    };
    await page.evaluate(({fingerprint, changes}) => localStorage.setItem(fingerprint, JSON.stringify(changes)), {fingerprint, changes: previousChanges});
    await page.evaluate(value => FinanceDashboard.loadData(value), data);
    assert.equal(await page.evaluate(() => FinanceDashboard.getFingerprint()), fingerprint, 'Full-refund exclusion must not invalidate previous review storage');
    assert.equal((await displayedIds(page)).length, 50);
    assert.deepEqual(await displayedIds(page), expectedOrder(active, 'time', 'desc').slice(0, 50));
    assert.equal(await page.locator('#transactionRows table').count(), 0);
    assert.equal(await page.locator('#canonicalCount').textContent(), '121');
    assert((await page.locator('#fullRefundNotice').textContent()).includes('2 组 / 4 条'));
    const expectedExpense = active.filter(row => row.kind === 'expense').reduce((sum, row) => sum + row.amount, 0);
    const summary = await page.evaluate(() => FinanceDashboard.getSummary());
    assert.equal(summary.expense, expectedExpense);
    assert.equal(summary.refund, 2_500);
    assert.equal(summary.flagged, 0, 'Old review=true on a removed row must not leak into pending totals');
    const categories = await page.evaluate(() => FinanceDashboard.getCategorySummary());
    assert.equal(categories.reduce((sum, group) => sum + group.net, 0), summary.net);
    assert.equal(categories.reduce((sum, group) => sum + group.count, 0), 120);
    assert.equal((await page.evaluate(() => FinanceDashboard.getEntitySummary('example-person'))).count, 121);
    assert.equal(await page.locator('#entityDirectory [data-entity="only-removed"]').count(), 0);
    assert.equal(await page.locator('#transactionRows .ledger-row').first().evaluate(row => row.getBoundingClientRect().height), 46);

    for (const size of [25, 50, 100, 200]) {
      await page.locator('#mainPageSize').selectOption(String(size));
      assert.equal((await displayedIds(page)).length, Math.min(active.length, size));
    }
    for (const field of ['amount', 'counterparty', 'spending_category', 'time']) {
      await page.locator('#mainSortField').selectOption(field);
      for (const direction of ['asc', 'desc']) {
        await page.locator('#mainSortDirection').selectOption(direction);
        assert.deepEqual(await displayedIds(page), expectedOrder(active, field, direction));
      }
    }
    await page.locator('#transactionHeader [data-sort="time"]').click();
    assert.equal(await page.locator('#mainSortDirection').inputValue(), 'asc');
    await page.locator('#mainSortField').selectOption('amount');
    await page.locator('#searchInput').fill('row-0');
    const filtered = active.filter(row => row.id.startsWith('row-0'));
    const pendingDownload = page.waitForEvent('download');
    await page.locator('#exportCsvBtn').click();
    const csv = await downloadText(await pendingDownload);
    const csvIds = csv.replace(/^\uFEFF/, '').split(/\r?\n/).slice(1).map(line => line.split(',')[0].slice(1, -1));
    assert.deepEqual(csvIds, expectedOrder(filtered, 'amount', 'asc'), 'CSV must honor current filtering and ordering');
    await page.locator('#searchInput').fill('');
    await page.locator('#mainPageSize').selectOption('50');
    await page.locator('#nextPage').click();
    assert.deepEqual(await displayedIds(page), expectedOrder(active, 'amount', 'asc').slice(50, 100));

    await page.locator('#entityDirectory [data-entity="example-person"]').click();
    assert.equal((await displayedIds(page, 'entity')).length, 50);
    await page.locator('#entityPageSize').selectOption('100');
    await page.locator('#entitySortField').selectOption('amount');
    await page.locator('#entitySortDirection').selectOption('desc');
    assert.deepEqual(await displayedIds(page, 'entity'), expectedOrder(active, 'amount', 'desc').slice(0, 100));
    assert.equal(await page.locator('#mainSortDirection').inputValue(), 'asc', 'The two lists sort independently');
    await page.locator('#entityNextPage').click();
    assert.equal((await displayedIds(page, 'entity')).length, 21);
    await page.locator('#monthFilter').selectOption('2026-02');
    assert.equal((await page.evaluate(() => FinanceDashboard.getEntitySummary('example-person'))).count, 121);
    await page.locator('#entityMonthFilter').selectOption('2026-03');
    assert((await displayedIds(page, 'entity')).every(id => active.find(row => row.id === id).month === '2026-03'));
    await page.locator('#resetFilters').click();

    await page.locator('#searchInput').fill('removed-purchase');
    assert.equal((await displayedIds(page)).length, 0);
    await page.locator('#mainShowRemoved').check();
    assert.equal((await displayedIds(page)).length, 2);
    await page.locator('#transactionRows [data-id="removed-purchase"] .ledger-detail button').click();
    assert((await page.locator('#editForm').textContent()).includes('已整组剔除'));
    assert.equal(await page.locator('#editKind').count(), 0);
    assert.equal(await page.locator('#modalBody .evidence').count(), 1);
    await page.locator('#closeRemovedDetail').click();
    const audit = {type: 'finance-review-corrections', version: 1, fingerprint, changes: previousChanges};
    await page.locator('#fileInput').setInputFiles({name: 'synthetic-old-review.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(audit))});
    await page.waitForFunction(() => document.getElementById('toast').textContent.includes('已恢复'));
    assert.equal((await page.evaluate(() => FinanceDashboard.getSummary())).expense, expectedExpense);
    assert.equal((await page.evaluate(() => FinanceDashboard.getSummary())).refund, 2_500);

    await page.locator('#entitySearch').fill('只有全退');
    await page.locator('#entityDirectory [data-entity="only-removed"]').click();
    assert.equal((await page.evaluate(() => FinanceDashboard.getEntitySummary('only-removed'))).count, 0);
    assert.equal(await page.locator('#entityShowRemoved').isChecked(), true);
    assert.equal((await displayedIds(page, 'entity')).length, 2);
    await page.locator('#closeEntity').click();

    await page.locator('#resetFilters').click();
    await page.locator('#searchInput').fill('purchase');
    await page.locator('#transactionRows [data-id="purchase"] .ledger-detail button').click();
    assert.equal(await page.locator('#editCategory').inputValue(), '保留原有审核');
    await page.locator('#editSpendingCategory').selectOption('住宿');
    await page.locator('#editForm button[type="submit"]').click();
    const changedCategories = await page.evaluate(() => FinanceDashboard.getCategorySummary());
    assert.equal(changedCategories.find(group => group.category === '住宿').net, 7_500, 'The partial refund inherits its original purchase category');
    assert.equal((await page.evaluate(() => FinanceDashboard.getSummary())).net, summary.net);
    await page.locator('#resetFilters').click();
    assert.equal(await page.locator('#transactionRows img').count(), 0);
    assert.equal(await page.evaluate(() => window.syntheticXss), undefined);
    if (process.env.FINANCE_SCREENSHOT_DIR) {
      fs.mkdirSync(process.env.FINANCE_SCREENSHOT_DIR, {recursive: true});
      await page.locator('#transactionPanel').screenshot({path: path.join(process.env.FINANCE_SCREENSHOT_DIR, 'dense-desktop.png')});
    }
    for (const width of [768, 390]) {
      await page.setViewportSize({width, height: 900});
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), width);
      assert.equal(await page.locator('#transactionRows .ledger-row').first().evaluate(row => row.getBoundingClientRect().height), 46);
    }
    if (process.env.FINANCE_SCREENSHOT_DIR) await page.locator('#transactionPanel').screenshot({path: path.join(process.env.FINANCE_SCREENSHOT_DIR, 'dense-mobile.png')});
    assert.deepEqual(errors, []);
    console.log('PASS: dense grid rows, mobile width, independent numeric/text/date sorting, stable IDs, page capacities, CSV ordering, full-refund audit locks, fingerprint and legacy review compatibility, active entity counts, purpose inheritance, and XSS-safe rendering.');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
