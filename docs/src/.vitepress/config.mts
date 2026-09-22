import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import { mathjaxPlugin } from './mathjax-plugin'
import { juliaReplTransformer } from './julia-repl-transformer'
import footnote from "markdown-it-footnote";
import path from 'path'

const mathjax = mathjaxPlugin()

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
}

const navTemp = {
  nav: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

const sidebarTemp = {
  sidebar: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

// DocumenterVitepress mirrors the whole `pages` tree into the navbar, a
// duplicate of the sidebar that can fill the bar edge to edge and push the
// GitHub link off screen. This folds named top-level entries into one menu.
//
//
// Empty on purpose: the seven top-level entries of `docs/pages.jl` fit the bar
// as they are. `API` is the one that must never go in here -- folding it puts
// the docstring reference behind a dropdown labeled something else, which reads
// to a visitor as the reference having been dropped from the manual. This is the
// place to fold `Validation` or `Differentiability` the day the top level grows.
const MORE: string[] = []

function curateNav(items: any[]): any[] {
  if (!MORE.length) return items
  const more = items.filter((i) => MORE.includes(i.text))
  const out = items.filter((i) => !more.includes(i))
  if (more.length) out.push({ text: 'Reference', items: more })
  return out
}

// VitePress renders a sidebar group expanded unless it says otherwise, and
// DocumenterVitepress hard-codes `collapsed: false`. `Validation` alone holds
// twelve pages, so the panel opens several screens tall; every group starts
// closed instead, and the group holding the current page still opens on its own.
function collapseGroups(node: any): any {
  if (Array.isArray(node)) return node.map(collapseGroups)
  if (node && typeof node === 'object') {
    const out: any = { ...node }
    if (Array.isArray(out.items)) {
      out.items = out.items.map(collapseGroups)
      out.collapsed = true
    }
    return out
  }
  return node
}

const nav = [
  ...curateNav(navTemp.nav as unknown as any[]),
  {
    component: 'VersionPicker'
  }
]

const sidebar = collapseGroups(sidebarTemp.sidebar as unknown as any)

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  lastUpdated: true,
  cleanUrls: true,
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // This is required for MarkdownVitepress to work correctly...
  head: [
    ['link', { rel: 'icon', href: 'REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON' }],
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    // ['script', {src: '/versions.js'], for custom domains, I guess if deploy_url is available.
    ['script', {src: `${baseTemp.base}siteinfo.js`}],
    // REPLACE_ME_DOCUMENTER_VITEPRESS_NOINDEX
  ],
  
  markdown: {
    codeTransformers: [juliaReplTransformer()],
    config(md) {
      md.use(tabsMarkdownPlugin);
      md.use(footnote);
      mathjax.markdownConfig(md);
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
  },
  vite: {
    plugins: [
      mathjax.vitePlugin,
    ],
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('REPLACE_ME_DOCUMENTER_VITEPRESS_DEPLOY_ABSPATH'),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    optimizeDeps: {
      exclude: [ 
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ], 
    }, 
    ssr: { 
      noExternal: [ 
        // If there are other packages that need to be processed by Vite, you can add them here.
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ], 
    },
  },
  themeConfig: {
    outline: 'deep',
    logo: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar,
    sidebarDrawer: 'REPLACE_ME_DOCUMENTER_VITEPRESS_SIDEBAR_DRAWER',
    editLink: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    socialLinks: [
      { icon: 'github', link: 'REPLACE_ME_DOCUMENTER_VITEPRESS' }
    ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    }
  }
})
