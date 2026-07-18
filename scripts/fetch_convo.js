(async () => {
  const sess = await fetch('/api/auth/session').then(r => r.json());
  const token = sess.accessToken;
  if (!token) return 'NO_TOKEN: ' + JSON.stringify(sess).slice(0, 300);
  const id = location.pathname.split('/c/')[1];
  const convo = await fetch('/backend-api/conversation/' + id, {
    headers: { 'Authorization': 'Bearer ' + token }
  }).then(r => r.json());
  return JSON.stringify(convo);
})()
