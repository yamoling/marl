import { z } from "zod";

export const MetricSelectionSchema = z.object({
    label: z.string(),
    category: z.string(),
});

export type MetricSelection = z.infer<typeof MetricSelectionSchema>;