/**
 * Load multimodal eval tasks from suite JSON.
 */

/**
 * @param {object} suite
 * @param {'image'|'audio'|null} [modality]
 */
export function loadMultimodalTasks(suite, modality = null) {
  const base = suite.base_url ?? '';

  const mapTask = (task, taskModality) => ({
    ...task,
    modality: task.modality ?? taskModality,
    media_url: task.media_url ?? (task.media_file ? `${base}${task.media_file}` : null),
  });

  if (suite.tasks?.length) {
    let tasks = suite.tasks.map((task) => mapTask(task, task.modality));
    if (modality) {
      tasks = tasks.filter((t) => t.modality === modality);
    }
    return tasks;
  }

  let tasks = [];
  if (!modality || modality === 'image') {
    tasks = tasks.concat((suite.image_tasks ?? []).map((t) => mapTask(t, 'image')));
  }
  if (!modality || modality === 'audio') {
    tasks = tasks.concat((suite.audio_tasks ?? []).map((t) => mapTask(t, 'audio')));
  }
  return tasks;
}

/**
 * @param {object[]} tasks
 */
export function summarizeMultimodalTasks(tasks) {
  const image = tasks.filter((t) => t.modality === 'image');
  const audio = tasks.filter((t) => t.modality === 'audio');
  return {
    total: tasks.length,
    image: image.length,
    audio: audio.length,
  };
}
