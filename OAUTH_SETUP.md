# GitHub OAuth Setup Guide

## Overview
This dashboard uses GitHub OAuth for authentication. Follow these steps to set up the OAuth token exchange proxy.

## Step 1: GitHub OAuth App Configuration (Already Done ✓)

Your OAuth app is already registered:
- **Client ID**: `Ov23liKmEJPqiVijfP1R`
- **Authorization callback URL**: Your GitHub Pages URL (https://superfastsimon.github.io/inspace-ai-automation/index.html)

## Step 2: Cloudflare Worker Setup (TO DO)

The proxy that exchanges auth codes for tokens runs on Cloudflare Workers.

### 2a. Install Wrangler
```bash
npm install -g wrangler
# or
npm install -D wrangler
```

### 2b. Authenticate with Cloudflare
```bash
wrangler login
```

### 2c. Add Your Client Secret to Environment Variables

**IMPORTANT**: Never commit your Client Secret to GitHub. Set it via Wrangler Secrets:

```bash
wrangler secret put CLIENT_SECRET
# Paste your GitHub OAuth Client Secret when prompted
```

To find your Client Secret:
1. Go to https://github.com/settings/developers
2. Select your "Saikou" OAuth App
3. Copy the **Client Secret** (it should start with `ghcs_`)

### 2d. Update Worker Code with Your Secret

In `src/index.js`, replace the placeholder:
```javascript
const CLIENT_SECRET = 'YOUR_GITHUB_OAUTH_CLIENT_SECRET';
```

With accessing it from environment:
```javascript
const CLIENT_SECRET = GITHUB_CLIENT_SECRET;
```

### 2e. Deploy the Worker
```bash
wrangler deploy
```

The worker will be available at: `https://gh-oauth-proxy.superfastsimon.workers.dev`

## Step 3: Test the OAuth Flow

1. Visit your dashboard: https://superfastsimon.github.io/inspace-ai-automation/index.html
2. Click "Continue with GitHub"
3. Authorize the app
4. You should be redirected back and see your GitHub user badge in the top-right

## Troubleshooting

### 404 After Authorization
- Check that the Cloudflare Worker is deployed and running
- Open browser DevTools (F12) → Network tab
- Check the request to `gh-oauth-proxy.superfastsimon.workers.dev`
- Verify it returns a JSON response with `access_token`

### "Invalid Client Secret" Error
- Make sure you've set the secret via `wrangler secret put CLIENT_SECRET`
- Verify the secret matches your GitHub OAuth app settings

### CORS Errors
- The worker includes CORS headers to allow requests from any origin
- If still blocked, check browser console for detailed error

## Security Notes

✅ **This is secure because:**
- Client Secret is stored only in Cloudflare's environment (never in code/repo)
- Token exchange happens server-to-server (not exposing secrets to browser)
- Only `read:user` scope is requested (minimal permissions)
- No sensitive data stored client-side

❌ **Do NOT:**
- Commit your Client Secret to GitHub
- Expose it in client-side code
- Use a weaker OAuth scope than necessary
