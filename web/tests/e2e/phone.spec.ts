import { expect, test } from "@playwright/test";

import { BASE, FIXTURE, ROUTES, routeApi } from "./helpers";

test("no route scrolls sideways at 390 pixels", async ({ page }) => {
  await routeApi(page, "down");
  for (const route of ROUTES) {
    await page.goto(`${FIXTURE}${BASE}${route}`);
    await page.waitForLoadState("networkidle");
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow, route).toBeLessThanOrEqual(0);
  }
});

test("on a phone the margin marks move above the answer", async ({ page }) => {
  await routeApi(page, "live");
  await page.goto(`${FIXTURE}${BASE}/desk/`);
  await page.getByTestId("question-picker").selectOption("rb-D-comparison-02");
  const script = page.locator('section[aria-labelledby="methods-title"] [data-testid="script"]').first();
  await expect(script.getByTestId("marks")).toBeVisible();
  const marks = await script.getByTestId("marks").boundingBox();
  const answer = await script.getByTestId("answer").boundingBox();
  expect(marks && answer && marks.y + marks.height <= answer.y + 1).toBe(true);
});
