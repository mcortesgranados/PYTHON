# FileName: 35_Internationalization_and_Localization.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Internationalization and Localization in Python

# Python provides support for internationalization (i18n) and localization (l10n) through the gettext module.

import gettext

# Example: Internationalization and Localization with gettext

# Initialize the gettext library with the desired language
lang = 'es'  # Spanish
gettext.bindtextdomain('messages', localedir='locales')
gettext.textdomain('messages')
trans = gettext.translation('messages', localedir='locales', languages=[lang], fallback=True)

# Translate a message
print(trans.gettext("Hello, World!"))
