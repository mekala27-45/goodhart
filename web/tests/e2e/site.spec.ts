import { expect, test } from "@playwright/test";

import { BASE, FIXTURE, REPO, ROUTES, STATEMENT, routeApi, watchErrors } from "./helpers";

for (const [name, origin] of [
  ["with the fixtures", FIXTURE],
  ["with the repository's data alone", REPO],
] as const) {
  test(`every route renders and has the statement in its footer, ${name}`, async ({ page }) => {
    await routeApi(page, "down");
    const errors = watchErrors(page);
    for (const route of ROUTES) {
      const response = await page.goto(`${origin}${BASE}${route}`);
      expect(response?.status(), route).toBe(200);
      await expect(page.locator("h1").first()).toBeVisible();
      await expect(page.locator("footer [data-statement]")).toHaveText(STATEMENT);
      await expect(page.getByTestId("pushback")).toBeVisible();
    }
    expect(errors).toEqual([]);
  });
}

test("the hero draws every panel named in its legend", async ({ page }) => {
  await page.goto(`${FIXTURE}${BASE}/`);
  const hero = page.getByTestId("hero");
  await expect(hero).toHaveAttribute("data-state", "ready");
  const named = await hero.locator("[data-legend-panel]").evaluateAll((els) => els.map((e) => e.getAttribute("data-legend-panel")));
  expect(named).toEqual(["honest", "lenient", "leaked", "lazy", "refusal", "random", "format"]);
  for (const arm of named) {
    const panel = hero.locator(`[data-panel="${arm}"]`);
    await expect(panel, `${arm} is drawn`).toHaveAttribute("data-drawn", "true");
    await expect(panel.locator('[data-series="paid"]')).toHaveCount(1);
    await expect(panel.locator('[data-series="confirmed"] circle').first()).toBeVisible();
  }
  // The defect arms that fired carry a rule at their step, and their predicted window as a band.
  await expect(hero.locator('[data-panel="lenient"] [data-rule="fired"]')).toContainText("fired 60");
  await expect(hero.locator('[data-panel="lenient"] [data-band="window"]')).toHaveCount(1);
  await expect(hero.locator('[data-panel="lenient"] [data-series="gap"]')).toHaveCount(1);
  // The title is written by the rule from the rows, and the source line says these are fixtures.
  await expect(hero.getByTestId("chart-message")).toContainText("lenient (D1) arm was paid");
  await expect(hero.getByTestId("source-line")).toContainText("end to end test fixtures, not results");
});

test("with no runs in the repository the hero says not yet run, in words", async ({ page }) => {
  await page.goto(`${REPO}${BASE}/`);
  const hero = page.getByTestId("hero");
  await expect(hero.getByTestId("chart-message")).toHaveText("No arm of the defect study has run yet: each panel shows only the window its prediction registered.");
  await expect(hero.getByTestId("panel-not-run")).toHaveCount(7);
  await expect(page.getByTestId("champion").getByTestId("not-yet-run")).toContainText("Not yet run.");
});

test("a chart's table and SQL toggles show its rows and the query behind them", async ({ page }) => {
  await page.goto(`${REPO}${BASE}/verifier/`);
  const chart = page.getByTestId("mutation-chart");
  await expect(chart).toHaveAttribute("data-state", "ready");
  await expect(chart.getByTestId("chart-message")).toContainText("near misses of the test golds");
  await chart.getByTestId("table-toggle").click();
  await expect(chart.getByTestId("data-table").locator("tbody tr")).toHaveCount(6);
  await chart.getByTestId("sql-toggle").click();
  await expect(chart.getByTestId("sql")).toContainText("read_json_auto('results/mutation.json')");
  await chart.getByTestId("sql-toggle").click();
  await expect(chart.locator('[data-testid="mutation-bars"] [data-row]')).toHaveCount(6);
});

test("the methods read the graded completions through DuckDB, with the manifest's intervals", async ({ page }) => {
  await page.goto(`${FIXTURE}${BASE}/methods/`);
  const chart = page.getByTestId("methods-chart");
  await expect(chart).toHaveAttribute("data-state", "ready");
  await expect(chart.locator('[data-testid="methods-rows"] [data-row]')).toHaveCount(6);
  await expect(chart.locator('[data-row="M4"] [data-interval]')).toHaveCount(1);
  await expect(chart.locator('[data-row="floor"] [data-mark="floor"]')).toHaveCount(1);
  await expect(chart.getByTestId("chart-message")).toContainText("M4 SFT then GRPO");
  await chart.getByTestId("sql-toggle").click();
  await expect(chart.getByTestId("sql")).toContainText("FROM grades");
  // The generated test set has no sampled answers in the fixtures; the readback test set has sixteen per question.
  await expect(page.getByTestId("coverage-chart")).toHaveAttribute("data-state", "missing");
  await page.locator("#coverage").getByRole("button", { name: "readback test" }).click();
  await expect(page.getByTestId("coverage-chart")).toHaveAttribute("data-state", "ready");
  await expect(page.locator('[data-testid="coverage-lines"] [data-line]')).toHaveCount(5);
});

test("every defect and control is printed beside its prediction, held or not", async ({ page }) => {
  await page.goto(`${FIXTURE}${BASE}/hacks/`);
  const cards = page.getByTestId("prediction-cards");
  await expect(cards.locator('[data-arm="lenient"] [data-testid="verdict"]')).toHaveText("held");
  await expect(cards.locator('[data-arm="lazy"] [data-testid="verdict"]')).toHaveText("did not hold");
  await expect(cards.locator('[data-arm="random"] [data-testid="verdict"]')).toHaveText("held");
  await expect(cards.locator('[data-arm="format"] [data-testid="verdict"]')).toHaveText("did not hold");
  const scripts = page.locator('[data-testid="arm-scripts"][data-arm="lenient"] [data-testid="script"]');
  await expect(scripts).toHaveCount(3);
  await expect(scripts.first().locator('[data-mark="reward"]').first()).toBeVisible();
});
