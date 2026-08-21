@'
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.ffbvision"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.ffbvision"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
    }
}
'@ | Set-Content ".\app\build.gradle.kts"