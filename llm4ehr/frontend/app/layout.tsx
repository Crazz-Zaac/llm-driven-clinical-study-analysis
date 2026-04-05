import type { Metadata } from 'next'
import { Source_Sans_3, JetBrains_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import { ThemeProvider } from '@/components/theme-provider'
import './globals.css'

const sourceSans = Source_Sans_3({ 
  subsets: ["latin"], 
  variable: "--font-source-sans",
  weight: ["300", "400", "500", "600", "700"]
});

const jetbrainsMono = JetBrains_Mono({ 
  subsets: ["latin"], 
  variable: "--font-jetbrains-mono"
});

export const metadata: Metadata = {
  title: 'LLM4EHR - AI-Powered Healthcare Research Assistant',
  description: 'Chat with LLM models for electronic health records research. Access Nature papers, upload PDFs, and configure local or Hugging Face models.',
  generator: 'v0.app',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${sourceSans.variable} ${jetbrainsMono.variable} font-sans antialiased`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="light"
          enableSystem
          storageKey="llm4ehr-theme"
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
        {process.env.NODE_ENV === 'production' && <Analytics />}
      </body>
    </html>
  )
}
