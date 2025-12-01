export async function fetchDashboardMock(baseUrl = '') {
  const url = baseUrl + '/api/v1/mock/dashboard';
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch dashboard mock');
  return await res.json();
}

export async function fetchDevEnvMock(baseUrl = '') {
  const url = baseUrl + '/api/v1/mock/dev_environment';
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch dev env mock');
  return await res.json();
}
