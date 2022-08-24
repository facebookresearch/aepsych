// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

window.addEventListener('load', function() {
    const cookieConsentContainer = document.querySelector(".cookie-container");
    const cookieConsentBtn = document.querySelector(".cookie-btn");
    const windowUrl = window.location.href
    
    if (windowUrl.includes("staticdocs")) {
        cookieConsentContainer.classList.remove("active");
        localStorage.setItem("cookieConsentDisplayed", true)
    }

    cookieConsentBtn.addEventListener("click", () => {
        cookieConsentContainer.classList.remove("active");
        localStorage.setItem("cookieConsentDisplayed", true)
    })

    setTimeout(()=> {
    const consentedToCookies = localStorage.getItem("cookieConsentDisplayed")

    if(!consentedToCookies){
        cookieConsentContainer.classList.add("active");
    }
    }, 2000)

    })
