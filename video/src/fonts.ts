import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import '@fontsource/space-grotesk/500.css';
import '@fontsource/space-grotesk/700.css';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/600.css';

/** Resolves once every face the canvas textures draw with is loaded. */
export const fontsReady = (): Promise<unknown> =>
  Promise.all([
    document.fonts.load('400 40px Inter'),
    document.fonts.load('500 40px Inter'),
    document.fonts.load('600 40px Inter'),
    document.fonts.load('700 40px Inter'),
    document.fonts.load('500 40px "Space Grotesk"'),
    document.fonts.load('700 40px "Space Grotesk"'),
    document.fonts.load('400 40px "JetBrains Mono"'),
    document.fonts.load('600 40px "JetBrains Mono"'),
  ]);
