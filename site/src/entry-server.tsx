// @refresh reload
import { createHandler, StartServer } from "@solidjs/start/server";

export default createHandler(() => (
  <StartServer
    document={({ assets, children, scripts }) => (
      <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <meta name="theme-color" content="#013A63" />
          <meta name="color-scheme" content="light" />
          <meta name="application-name" content="Fenua Intelligence" />
          <meta name="apple-mobile-web-app-capable" content="yes" />
          <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
          <meta name="apple-mobile-web-app-title" content="Fenua AI" />
          <meta name="mobile-web-app-capable" content="yes" />
          {/* DNS prefetch + preconnect for image CDNs */}
          <link rel="dns-prefetch" href="//e0.365dm.com" />
          <link rel="dns-prefetch" href="//assets.goal.com" />
          <link rel="dns-prefetch" href="//digitalhub.fifa.com" />
          <link rel="preconnect" href="https://e0.365dm.com" crossorigin="" />
          <link rel="preconnect" href="https://assets.goal.com" crossorigin="" />
          <link rel="preconnect" href="https://digitalhub.fifa.com" crossorigin="" />
          <link rel="icon" type="image/png" sizes="192x192" href="/icons/icon-192.png" />
          <link rel="manifest" href="/manifest.json" />
          <link rel="apple-touch-icon" href="/icons/icon-512.png" />
          {assets}
        </head>
        <body>
          <div id="app">{children}</div>
          {scripts}
        </body>
      </html>
    )}
  />
));
