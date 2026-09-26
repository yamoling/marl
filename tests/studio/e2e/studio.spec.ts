import { expect, test, type Page } from "@playwright/test";

async function loadHealthy(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: "Open Default" }).click();
  await expect(page.getByRole("heading", { name: "Welcome to MARL Studio" })).toBeVisible();
  await page.getByRole("button", { name: "Add experiments" }).last().click();
  await page.getByRole("textbox", { name: "Search experiments" }).fill("id=healthy");
  await expect(page.getByRole("checkbox", { name: "Select healthy" })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "Select light" })).toHaveCount(0);
  await page.getByRole("checkbox", { name: "Select healthy" }).check();
  await page.getByRole("button", { name: "Load 1 selected" }).click();
  await expect(page.getByRole("region", { name: "Workspace" })).toContainText("1 experiment");
}

test("create, rename, and open a workspace using the shared logs directory", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Add a new workspace" }).click();
  await page.getByRole("textbox", { name: "Workspace name" }).fill("Research");
  await page.getByRole("button", { name: "Create" }).click();
  await expect(page.locator(".workspace-card")).toHaveCount(2);
  const card = page.locator(".workspace-card").last();
  await card.getByRole("button", { name: "Rename Research" }).click();
  await card.getByRole("textbox", { name: "Rename Research" }).fill("Renamed");
  await card.getByRole("textbox", { name: "Rename Research" }).press("Enter");
  await page.getByRole("button", { name: "Open Renamed" }).click();
  await page.getByRole("button", { name: "Add experiments" }).last().click();
  await expect(page.getByRole("checkbox", { name: "Select healthy" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("link", { name: "Choose workspace" })).toContainText("Renamed");
  await page.getByRole("link", { name: "Choose workspace" }).click();
  await expect(page.getByRole("heading", { name: "Renamed" })).toBeVisible();
});

test("plot train and test together, add another plot, maximize and minimize", async ({ page }) => {
  await loadHealthy(page);
  const workspace = page.getByRole("region", { name: "Workspace" });
  await expect(page.getByRole("complementary", { name: "Fields" })).toContainText("exit_rate");
  await workspace.getByRole("button", { name: /Train vs test/ }).click();
  const comparison = workspace.getByRole("article", { name: "Plot Train vs test" });
  await expect(comparison.getByRole("button", { name: /^healthy · train\/score-0/ })).toBeVisible();
  await expect(comparison.getByRole("button", { name: /^healthy · test\/score-0/ })).toBeVisible();
  await expect(comparison).toContainText("2 series");
  const lines = comparison.locator("svg .mc-data path.mc-center");
  await expect(lines).toHaveCount(2);
  await expect(lines.nth(0)).toHaveAttribute("d", /^M/);
  await expect(lines.nth(1)).toHaveAttribute("d", /^M/);

  await page
    .getByRole("complementary", { name: "Fields" })
    .locator('[data-table="test"]')
    .getByRole("button", { name: /gems_collected/ })
    .click();
  await page.getByRole("dialog", { name: "Add test/gems_collected to…" }).getByRole("button", { name: "New plot" }).click();
  const second = workspace.getByRole("article", { name: "Plot gems_collected (test)" });
  await expect(second).toContainText("1 series");
  await expect(workspace).toContainText("2 plots");

  await comparison.getByRole("button", { name: "Maximize" }).click();
  const overlay = page.getByRole("dialog", { name: "Train vs test" });
  await expect(overlay.getByRole("article", { name: "Plot Train vs test" })).toContainText("2 series");
  await overlay.getByRole("button", { name: "Restore" }).click();
  await expect(overlay).toHaveCount(0);
  await comparison.getByRole("button", { name: "Minimize" }).click();
  await expect(comparison.getByRole("button", { name: "Expand" })).toBeVisible();
  await comparison.getByRole("button", { name: "Expand" }).click();
  await expect(comparison.getByRole("button", { name: "Minimize" })).toBeVisible();
  await expect(second).toContainText("1 series");
});

test("inspect the experiment drawer and fixture episodes", async ({ page }) => {
  await loadHealthy(page);
  await page.getByRole("button", { name: "healthy Unload healthy", exact: true }).click();
  const drawer = page.getByRole("dialog", { name: "healthy" });
  await expect(drawer.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
  await drawer.getByRole("tab", { name: /Runs/ }).click();
  await expect(drawer).toContainText("3 runs");
  await drawer.getByRole("tab", { name: "Parameters" }).click();
  await expect(drawer.getByRole("tab", { name: "Parameters" })).toHaveAttribute("aria-selected", "true");
  await drawer.getByRole("button", { name: "Episodes" }).click();

  const episodes = page.getByRole("dialog", { name: "Episodes" });
  await expect(episodes.getByRole("heading", { name: "Episodes at step 10,000" })).toBeVisible();
  await expect(episodes.getByRole("combobox", { name: "Experiment" })).toHaveValue("healthy");
  await expect(episodes.getByRole("button", { name: /test #/ })).toHaveCount(6);
  await episodes.getByRole("button", { name: "Previous test step" }).click();
  await expect(episodes.getByRole("heading", { name: "Episodes at step 9,000" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(episodes).toHaveCount(0);
  await expect(drawer).toBeVisible();
  await drawer.getByRole("button", { name: "Close" }).click();
});

test("compare parameters from two disposable fixture experiments", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Open Default" }).click();
  await page.getByRole("button", { name: "Add experiments" }).last().click();
  await page.getByRole("checkbox", { name: "Select healthy" }).check();
  await page.getByRole("checkbox", { name: "Select light" }).check();
  await page.getByRole("button", { name: "Load 2 selected" }).click();
  await expect(page.getByRole("region", { name: "Workspace" })).toContainText("2 experiments");
  await page.getByRole("button", { name: "Compare parameters" }).click();
  await expect(page.getByRole("heading", { name: "2 experiments" })).toBeVisible();
  await expect(page.getByText(/parameters differ/)).toBeVisible();
});
