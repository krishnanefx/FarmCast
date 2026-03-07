import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");
const outputDir = path.join(repoRoot, "frontend", "data");

const atlasFiles = [
  "land-110m.json",
  "countries-110m.json",
  "land-50m.json",
  "countries-50m.json",
];

async function copyAtlasFile(moduleDir, fileName) {
  const source = path.join(moduleDir, fileName);
  const target = path.join(outputDir, fileName);
  await fs.copyFile(source, target);
  return target;
}

async function main() {
  const pkgPath = require.resolve("world-atlas/package.json");
  const moduleDir = path.dirname(pkgPath);

  await fs.mkdir(outputDir, { recursive: true });

  const copiedFiles = [];
  for (const fileName of atlasFiles) {
    const target = await copyAtlasFile(moduleDir, fileName);
    copiedFiles.push(target);
  }

  const manifestPath = path.join(outputDir, "manifest.json");
  await fs.writeFile(
    manifestPath,
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        source: "world-atlas",
        files: atlasFiles,
      },
      null,
      2
    )
  );

  process.stdout.write(
    [
      "Map data loaded for HPC runtime.",
      `Source module: ${moduleDir}`,
      `Output dir: ${outputDir}`,
      `Files: ${atlasFiles.join(", ")}`,
    ].join("\n") + "\n"
  );
}

main().catch((error) => {
  process.stderr.write(`Failed to load map data: ${error.message}\n`);
  process.exit(1);
});
