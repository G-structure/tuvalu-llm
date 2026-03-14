// @refresh reload
import { createHandler, StartServer } from "@solidjs/start/server";

export default createHandler(() => (
  <StartServer
    document={({ assets, children, scripts }) => (
      <html lang="tvl">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <meta name="theme-color" content="#013A63" />
          {/* DNS prefetch + preconnect for image CDNs */}
          <link rel="dns-prefetch" href="//e0.365dm.com" />
          <link rel="dns-prefetch" href="//assets.goal.com" />
          <link rel="dns-prefetch" href="//digitalhub.fifa.com" />
          <link rel="preconnect" href="https://e0.365dm.com" crossorigin="" />
          <link rel="preconnect" href="https://assets.goal.com" crossorigin="" />
          <link rel="preconnect" href="https://digitalhub.fifa.com" crossorigin="" />
          <link rel="icon" href="/favicon.ico" />
          <link rel="manifest" href="/manifest.json" />
          <link rel="apple-touch-icon" href="/icons/icon-192.png" />
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
