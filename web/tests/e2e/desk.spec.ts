import { expect, test } from "@playwright/test";

import { apiFixture, BASE, FIXTURE, routeApi } from "./helpers";

test("a method's answer prints as a script with its marks in the margin", async ({ page }) => {
  await routeApi(page, "live");
  await page.goto(`${FIXTURE}${BASE}/desk/`);
  await expect(page.getByTestId("desk-source")).toHaveAttribute("data-source", "live");
  await page.getByTestId("question-picker").selectOption("rb-D-comparison-02");
  const scripts = page.locator('section[aria-labelledby="methods-title"] [data-testid="script"]');
  await expect(scripts).toHaveCount(5);
  const first = scripts.first();
  await expect(first.getByTestId("answer")).toContainText("```sql");
  // Red for what the reward paid, green for what the checks confirmed: a word and a number each.
  await expect(first.locator('[data-mark="reward"]').first()).toHaveText(/^honest (0|1)\.\d\d$/);
  await expect(first.locator('[data-mark="checks"]').filter({ hasText: "confirmed" })).toHaveText(/^confirmed [01]$/);
});

test("a live grade renders with what each reward pays and what the twins confirm", async ({ page }) => {
  const seen = await routeApi(page, "live");
  await page.goto(`${FIXTURE}${BASE}/desk/`);
  await expect(page.getByTestId("desk-source")).toHaveAttribute("data-source", "live");
  await page.getByTestId("question-picker").selectOption("rb-D-comparison-02");
  await page.getByTestId("sql-editor").fill("SELECT cbsa_title AS label, max(value_anchored) AS value FROM doorstep.mart_metro_year GROUP BY 1");
  await page.getByTestId("mark").click();
  const grade = page.getByTestId("grade");
  await expect(grade).toHaveAttribute("data-source", "live");
  const marks = grade.getByTestId("marks");
  for (const text of ["honest 1.00", "lenient 1.00", "lazy 1.00", "refusal 1.00"]) await expect(marks.locator('[data-mark="reward"]').filter({ hasText: text })).toHaveCount(1);
  await expect(marks.locator('[data-mark="checks"]').filter({ hasText: "twins 4 of 4" })).toHaveCount(1);
  await expect(grade.getByTestId("data-table").locator("tbody tr")).toHaveCount(2);
  const fixture = apiFixture<{ grade_id: string }>("grade");
  await expect(grade).toContainText(fixture.grade_id);
  // The page posted the question and the query, and never sent HEAD.
  const post = seen.find((s) => s.method === "POST");
  expect(JSON.parse(post?.body ?? "{}")).toEqual({ question_id: "rb-D-comparison-02", sql: "SELECT cbsa_title AS label, max(value_anchored) AS value FROM doorstep.mart_metro_year GROUP BY 1" });
  expect(seen.filter((s) => s.method === "HEAD")).toEqual([]);
  expect(seen.filter((s) => s.url.endsWith("/health")).length).toBeGreaterThanOrEqual(1);
});

test("when the API does not answer, the recorded session plays and says it is recorded", async ({ page }) => {
  const seen = await routeApi(page, "down");
  await page.goto(`${FIXTURE}${BASE}/desk/`);
  const source = page.getByTestId("desk-source");
  await expect(source).toHaveAttribute("data-source", "recorded", { timeout: 40_000 });
  await expect(source).toContainText("Recorded.");
  await expect(source).toContainText("nothing below is graded now");
  // Two probes of /health, and nothing else, before the recording took over.
  expect(seen.filter((s) => s.url.endsWith("/health"))).toHaveLength(2);
  const tries = page.getByTestId("try");
  await expect(tries).toHaveCount(5);
  await tries.filter({ hasText: "print the number" }).click();
  const grade = page.getByTestId("grade");
  await expect(grade).toHaveAttribute("data-source", "recorded");
  await expect(grade.locator('[data-mark="reward"]').filter({ hasText: "honest 1.00" })).toHaveCount(1);
  await expect(grade.locator('[data-mark="checks"]').filter({ hasText: /^twins 0 of 4$/ })).toHaveCount(1);
  await expect(grade.getByTestId("flags")).toContainText("literal only");
  // A query the recording does not hold is refused in words, not graded.
  await page.getByTestId("sql-editor").fill("SELECT 1");
  await page.getByTestId("mark").click();
  await expect(page.getByTestId("grade-error")).toContainText("the recording has no grade for this query");
});
